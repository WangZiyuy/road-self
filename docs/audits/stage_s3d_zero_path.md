# Stage S3D Zero Path

Formal run code SHA: `68566e6319d59bf9726beda656c6abbe372b5be4`.

The audit classified the earlier zero-path discrepancy as `MULTIPLE_CAUSES`. Stage S3D replaced that path with a strict zero-preserving residual contract:

- the adapter has no bias or normalization offset;
- only the raster stream is transformed;
- support and valid masks gate the projected residual;
- zero raster satisfies `F(x, 0) = x` exactly, including after an optimizer step;
- fusion affects the road path only;
- the junction path remains image-only;
- no raw raster or raster embedding is routed directly to anchor inference.

The corrected production preflight passed zero identity, zero/null road-logit equality, junction invariance, and finite-gradient checks. The strict adapter contract, road-only routing contract, and junction parity contract all passed.

All formal CUDA evidence was produced on REMOTE_TRAINING_SERVER.
