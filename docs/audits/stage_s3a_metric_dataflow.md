# Stage S3A metric dataflow

`final_metrics` is classified as **LAST_TRAIN_BATCH_METRICS**. It is the last training batch only: logits were produced in `model.train()`, the optimizer then stepped, and the pre-update output was measured without cross-batch accumulation. It is not validation evidence.

`best_metrics` uses `model.eval()` and `torch.no_grad()` over the frozen 8-batch / 16-sample validation plan. Historical segmentation and anchor comparison files came from the overwritten latest evaluation; historical graph comparison used per-run best checkpoints.

The frozen segmentation composite is `(road F1 + road IoU + junction F1) / 3`. Full machine-readable provenance is in `artifacts/stage_s3a_metric_provenance.json`.
