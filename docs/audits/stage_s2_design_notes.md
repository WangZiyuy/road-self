# Stage S2 isolated raster-seg-only design notes

## Outcome

Stage S2 implements one trajectory-raster meaning and one model path. The
authoritative Xi'an raster is converted at every data boundary with
`traj_binary = (traj_raw > 0).astype(float32)`. Values such as 128 and 255 are
therefore density/intensity observations of the same presence class; they are
never supplied to the Stage S2 network as separate values or classes.

The formal mode is `TRAJ.MODE: raster_seg_only`. It uses an image-only Res2Net
backbone and adds a 106,984-parameter raster encoder/fusion module only in front
of the road and junction segmentation heads. It does not construct the legacy
coordinate-sequence Transformer, `fuse_module_traj`, trajectory ranking, or
DSFNet.

## Canonical Xi'an input

The user-supplied authority supersedes the Stage S1 unknown-semantics blocker:

- logical registered data extent: H=5000, W=4300;
- canonical full canvas: H=W=8192;
- valid upper-left extent: W=4300, H=5000;
- full-canvas padding: 3892 columns on the right and 3192 rows on the bottom;
- canonical values: exactly 0 (no support) and 1 (trajectory present);
- valid-mask values: exactly 0 and 1, with padding marked invalid;
- current experiment scope: Xi'an only.

The local full-canvas PNG and its derived 4096 upper-left tile live below
`data_self/input/`, are ignored by Git, and are recorded by checksum in
`artifacts/stage_s2_canonical_raster_manifest.json`. The actual raster, aerial
image, dataset, and derived tile are not commit candidates.

Training window extraction retains the repository's legacy X,Y internal array
convention only inside the TileCache adapter. It applies the canonical `> 0`
conversion before CHW conversion. Full-canvas inference uses the same canonical
loader in H,W order, then performs one explicit transpose at the existing
inference adapter so aerial and raster crops retain identical indices.

## Model path

The formal forward path is:

```text
aerial RGB
  -> image-only Res2Net stages 1..5
  -> stage_fuse_img -------------------------------> anchor recursion
                    \
traj_binary           -> raster encoder -> residual fusion -> stage_fuse_seg
traj_valid_mask      /                                  |
                                                        +-> road head -> road_fts
                                                        +-> junc head -> junc_fts
                                                                    |
                                       road_fts / junc_fts ----------+
                                                                    v
                                                              anchor recursion
```

`TrajectoryRasterEncoder` performs two stride-2 convolutions and one lightweight
depthwise/projection block, producing `[B,32,H/4,W/4]`.
`SegmentationOnlyRasterFusion` concatenates the image segmentation feature,
projected raster feature, and downsampled valid mask. Its final 1x1 residual
projection is zero-initialized, so the initial `stage_fuse_seg` is exactly equal
to `stage_fuse_img` while remaining trainable from segmentation losses.

The original anchor Hourglass receives its original image feature, road/junction
features, walked path, and recursive placeholder. Its input width is unchanged;
no raster channel or embedding is added. Original multi-scale decoder stages and
recursive feedback remain unchanged.

## Gradient policy

`TRAJ.RASTER.ANCHOR_GRAD_TO_SEG` controls only the features passed from the
segmentation heads to the anchor:

- `false`: anchor receives `road_fts.detach()` and `junc_fts.detach()`. Anchor
  loss cannot train the raster encoder/fusion or segmentation heads; road and
  junction losses still train them.
- `true`: anchor receives live road/junction features and may jointly train the
  segmentation path.

The detach profile is the stricter causal experiment for asking whether improved
segmentation inference alone helps anchor prediction. The joint profile is a
separate follow-up comparison; both retain the same forward isolation boundary.

## Train and inference wiring

`train.py`, `infer.py`, and `OSMDataset` now distinguish `use_raster` from
`use_sequence`. In `raster_seg_only` mode they load raster plus valid mask, but
the region sequence loader and sequence padding/normalization return immediately
without being called. `RPNet` is constructed with
`enable_raster_segmentation=True` and `enable_trajectory_modules=False`.

Road, junction, full-resolution anchor, and low-resolution anchor outputs remain
logits. Training continues to use `binary_cross_entropy_with_logits`; sigmoid is
applied only at metrics, visualization, or inference boundaries.

## Validation performed

CPU smoke validation used the trusted canonical Xi'an full-canvas raster and a
read-only aligned Xi'an aerial canvas for one 128x128 crop. Image-only,
raster-detach, and raster-joint variants each completed a forward/backward step
with finite loss. The two raster variants gave every raster module parameter a
finite gradient and at least one non-zero gradient. A segmentation inference
crop returned finite road and junction logits without any trajectory sequence.

The 8192x8192 raster and valid extent were validated as loader inputs. A complete
8192 neural inference sweep was not run in S2: it needs the intended trained S2
checkpoint and production GPU budget, whereas this stage's required smoke infer
is tile-level.

## Configuration profiles

- `configs/seg_raster_isolated.yml`: joint anchor-to-seg gradient.
- `configs/seg_raster_isolated_detach.yml`: detached anchor-to-seg gradient.

Both are Xi'an-only profiles, select `MODEL: origin`, set
`INPUT_SEMANTICS: binary_presence`, enable the valid mask, and explicitly disable
the trajectory sequence path.
