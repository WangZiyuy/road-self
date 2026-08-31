# Stage S3D Input Dependence

Formal run code SHA: `68566e6319d59bf9726beda656c6abbe372b5be4`.

The forensic execution completed, but the scientific input-specificity gate failed:

- `CURRENT_RASTER_INPUT_DEPENDENCE = FAIL`
- the raster residual changes when the input changes: `true`
- aligned raster is strictly best across the declared controls: `false`

The controls were strong enough for this conclusion. The 512-pixel zero-fill shift reduced spatial correspondence, and the deterministic permutation preserved raster density while breaking location. Nevertheless, the aligned input did not outperform zero, shifted, and permuted controls under the predeclared comparison.

This distinguishes a functioning input-dependent adapter from useful spatially specific trajectory evidence: the former was observed, while the latter was not.

All formal CUDA evidence was produced on REMOTE_TRAINING_SERVER.
