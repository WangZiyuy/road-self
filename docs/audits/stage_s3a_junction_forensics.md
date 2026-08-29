# Stage S3A junction forensics

Root-cause classes: `CALIBRATION_COLLAPSE, CLASS_IMBALANCE, LATE_TRAINING_OVERFIT, LOSS_DOMINATED_BY_BACKGROUND, THRESHOLD_TOO_HIGH`.

The model/evaluator contract passes: junction outputs are logits, one sigmoid is applied at the metric boundary, evaluation is `eval/no_grad`, and output/target are 64×64. F1=0 at threshold 0.3 means no thresholded hits; it does not mean logits or probabilities are numerically all zero. AUPRC, quantiles, calibration and a shared diagnostic threshold sweep are retained in the JSON. Per-run tuned thresholds are not used for causal claims.
