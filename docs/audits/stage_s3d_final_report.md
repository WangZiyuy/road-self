# Stage S3D Final Report

S3D base SHA: `2271a83a563c39f3d971df9a6342ad6c8a464a78`

Formal run code SHA: `68566e6319d59bf9726beda656c6abbe372b5be4`

Remote verification passed with 389 repository tests and 8 targeted tests. Preparation, forensic execution, corrected production preflight, all five controlled runs, strict zero-preserving fusion, road-only routing, and junction parity completed successfully.

The decisive spatial-causal result was negative: aligned binary raster was worse than image-only, zero, shifted, and permuted controls at the common 40960-sample comparison point. Consequently:

- segmentation spatial-causal gate: `FAIL`
- frozen anchor evaluation: `NOT_EXECUTED_BY_GATE`
- resource-capped graph evaluation: `NOT_EXECUTED_BY_GATE`
- segmentation multi-seed: `NO_GO`
- end-to-end multi-seed: `NO_GO`
- raster branch decision: `STOP_AFTER_NEGATIVE_CONTROL_RESULT`

The strict N0/N2 null-parity gate also remained `FAIL` because the gradient checksum diverged despite equality of predictions, features, shared parameter state, and metrics. This is retained as a protocol risk, not hidden by the matching observable outputs.

Two superseded code attempts were invalidated before any formal adaptation optimizer step: `a0aa632dc6a61c74fc7849806c566dde06d3206f` and `ff61ec1c343670da65b02e019d9ee6624e75d004`. GPU contention was recorded, and wall-clock timing under contention was excluded from model-quality evidence.

All formal CUDA evidence was produced on REMOTE_TRAINING_SERVER.
