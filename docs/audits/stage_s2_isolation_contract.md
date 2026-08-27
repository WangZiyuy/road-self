# Stage S2 raster-to-anchor isolation contract

## Required causal boundary

The raster may affect anchor predictions only through road and junction
segmentation features:

```text
allowed:
traj_binary -> stage_fuse_seg -> road_fts/junc_fts -> anchor

forbidden:
traj_binary -> anchor
traj_feature -> anchor
raster gate or embedding -> anchor decoder
stage_fuse_seg -> anchor image-feature input
```

The image backbone accepts only the aerial RGB tensor. Its five image stages and
`stage_fuse_img` are computed before any raster operation. The anchor retains
the original `stage_fuse_img` and original multi-scale image stages. Only the
road and junction heads receive `stage_fuse_seg`.

## Enforced interfaces

`RPNet` has separate construction flags:

- `enable_trajectory_modules` is only for `legacy_current`;
- `enable_raster_segmentation` is only for `raster_seg_only`;
- constructing both is rejected;
- `raster_seg_only` with `MODEL=DSFNet` is rejected;
- missing or spatially misaligned raster input is rejected.

For the original one-target anchor, the Hourglass input remains 257 channels:
128 image fusion + 64 road + 64 junction + 1 walked path. Recursive placeholder
channels are added only for later targets exactly as before. The raster encoder
output is stored only as an optional diagnostic tensor and is never supplied to
an anchor module.

## Gradient boundary

With `ANCHOR_GRAD_TO_SEG=false`, the two segmentation feature tensors are
detached at the anchor boundary. An anchor-only backward pass produces no raster
fusion gradients. Segmentation losses still pass through the raster fusion.

With `ANCHOR_GRAD_TO_SEG=true`, an anchor-only backward pass reaches the raster
fusion through road/junction features. It still has no direct raster-to-anchor
forward edge.

## Evidence

The Stage S2 isolation tests provide four independent checks:

1. construction checks prove no Transformer, `fuse_module_traj`, or DSF module
   is registered in `raster_seg_only`;
2. a forward pre-hook observes the unchanged anchor Hourglass input width;
3. after replacing road/junction feature heads with fixed outputs, changing the
   raster from all-zero to all-one leaves anchor output exactly unchanged;
4. anchor-only backward produces no raster gradient in detach mode and finite,
   partly non-zero raster gradients in joint mode.

An origin parity test resets the same RNG seed, compares every pre-existing
state-dict tensor, and compares road, junction, anchor, and anchor-low-resolution
outputs. `TRAJ.MODE=none` registers no raster module and remains exactly equal to
the original image-only path.

Machine-readable evidence is in
`artifacts/stage_s2_isolation_audit.json` and test results are in
`artifacts/stage_s2_test_results.json`.
