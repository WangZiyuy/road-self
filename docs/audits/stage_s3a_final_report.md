# Stage S3A final report

Stage S3A completed a read-only post-training audit of the immutable S3 checkpoints. There were 120 historical validation log points but only 12 retained checkpoint files (`best` and `latest` for six runs); Protocol C is therefore **UNAVAILABLE_MISSING_CHECKPOINT**, not reconstructed.

- Metric reference gate: **PASS**
- Final metrics: **LAST_TRAIN_BATCH_METRICS**
- Junction: `CALIBRATION_COLLAPSE, CLASS_IMBALANCE, LATE_TRAINING_OVERFIT, LOSS_DOMINATED_BY_BACKGROUND, THRESHOLD_TOO_HIGH`
- Anchor specificity: **INCONCLUSIVE**
- Pixel parity: **PASS**
- Evaluation determinism: **PASS**
- C1-best computational feasibility: **FAIL**
- C1-best graph expansion: **PATHOLOGICAL**
- C1-best graph metrics: **NOT_AVAILABLE_INCOMPLETE_RUN**
- Segmentation-only multi-seed: **NO_GO**
- End-to-end multi-seed: **NO_GO**
- Multi-seed decision: **NO_GO**

Checkpoint selection finding: segmentation-composite best checkpoint is not graph-stable; C1-latest terminates normally while C1-best exhibits pathological expansion.

The original Stage S3 artifacts remain unchanged. Stage S3A narrows or supersedes claims in a separate reconciliation artifact. The failed C1-best run is preserved as failed evidence and is never represented as `PASS`.
