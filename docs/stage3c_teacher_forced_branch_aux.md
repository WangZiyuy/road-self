# Stage 3C teacher-forced branch auxiliary training

Stage 3C trains the Stage 3B unordered immediate-branch head while keeping
road_self's RPNet and graph-growth state machine unchanged. Branch predictions
remain diagnostic auxiliary outputs: they are never passed to
`map_to_coordinate` or `Path.push`.

## Inputs and trainable modules

The frozen image-only RPNet supplies `feature_maps["stage_fuse"]`. A lightweight
convolution projects the 64 x 64 walked-path raster and fuses it at the
corresponding image-token positions. `GraphStateEncoder` encodes the incoming
direction and explored edges. `TrajectoryFragmentEncoder` independently
encodes the 64 fragments selected by explicit
`bounded_near_diverse(K=64, prepool_multiplier=8, near_fraction=0.5)`
compression.

Only these modules are optimized:

- `TrajectoryFragmentEncoder`;
- `GraphStateEncoder`;
- the walked-path projection in `MultiModalBranchQueryDecoder`;
- `MultiModalBranchQueryDecoder`.

RPNet runs in `eval` mode under `torch.no_grad()`. The auxiliary checkpoint does
not duplicate RPNet parameters; it records the exact image-only checkpoint
reference and saves the three auxiliary module state dictionaries plus the
optimizer state.

Because every crop and the frozen RPNet are constant, formal training can
precompute `stage_fuse` once into a volatile float32 CPU cache. This cache is
not written into the dataset, is not part of a checkpoint, and preserves the
same RPNet output precision. Setting
`STAGE3C.TRAINING.PRECOMPUTE_RPNET_FEATURES=False` restores direct per-batch
feature extraction.

## Dataset preparation

The cache builder follows the teacher-forced state sequence:

```text
Path.pop_state
-> make_path_input
-> get_target_poses
-> build_graph_state
-> build_immediate_branch_targets
-> Path.push(follow_target)
```

The default split uses disjoint 2048 x 2048 subtiles and additionally verifies
that no exact center coordinate occurs in both splits. Samples are stored in
fixed-shape compressed NPZ shards with `allow_pickle=False`; no Python object
arrays or pickle dataset records are used.

```bash
python scripts/prepare_stage3c_branch_dataset.py \
  --config configs/stage3c_branch_aux.yml \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --output-dir data_self/stage3c_branch_dataset \
  --overwrite
```

The default configuration prepares 2048 training states and 512 validation
states. Each sample contains the aerial crop, walked path, padded graph state,
immediate branch targets, fixed trajectory tensors and explicit masks.

## Mandatory overfit gate and formal training

Formal training requires a compatible image-only checkpoint and fails instead
of falling back to random RPNet weights. The `train` mode first runs the same
32-sample overfit gate as `sanity` mode and starts formal training only if total
loss, endpoint error and direction error all satisfy the configured reduction
thresholds.

```bash
python train_branch_aux.py \
  --config configs/stage3c_branch_aux.yml \
  --mode sanity \
  --device cuda

python train_branch_aux.py \
  --config configs/stage3c_branch_aux.yml \
  --mode train \
  --device cuda
```

Trajectory modality dropout is applied per sample during formal training. A
dropped sample has an all-false fragment mask and therefore naturally uses the
image, walked path and graph-state modalities.

## Validation and ablation

Hungarian matching uses endpoint L1 and continuous direction cosine costs.
Validation reports:

- branch precision, recall and F1;
- exact branch-count accuracy;
- mean and median endpoint error in pixels;
- mean and median direction error in degrees;
- missed-branch and extra-branch rates;
- duplicate-query pair and duplicate-node ratios.

Every formal epoch compares `full` and `no_trajectory`. The best checkpoint is
selected by full-modality validation F1, then evaluated with:

- `full`: image + walked path + graph state + trajectory;
- `no_trajectory`: image + walked path + graph state;
- `trajectory_graph`: trajectory + graph state, with image and walked-path
  context suppressed.

Attention weights remain diagnostic attention allocation and are not treated
as calibrated trajectory-selection probabilities.

```bash
python train_branch_aux.py \
  --config configs/stage3c_branch_aux.yml \
  --mode evaluate \
  --checkpoint data_self/stage3c_branch_aux/checkpoints/stage3c_aux.best.pth.tar
```

Outputs include JSON training curves, TensorBoard events, PNG curves, strict
auxiliary checkpoints, run metadata and final ablation metrics.

## Executed Xi'an result (2026-07-24)

The executed run used the strictly loaded image-only checkpoint:

```text
/home/wangziyu/VecRoad-master/data/ckpt/vecroad.pth.tar
SHA256 498abc76e4ea461040b2b4ce69dc3d896cb975e93aae58815fe76443f6acf7c3
```

The checkpoint contains an official image-only VecRoad state dictionary and
was loaded with `strict=True`. RPNet was frozen for the entire auxiliary run.

The teacher-forced cache contains 2048 training states and 512 validation
states in disjoint subtiles, with zero cross-split center overlap. Cache
construction took 942.6 seconds and produced 385 MB of NPZ shards. The training
and validation states had, respectively, 1065.8 and 750.4 full recalled
fragments on average before bounded compression.

The mandatory 32-sample sanity check passed after 400 iterations:

- total loss reduction: 82.63%;
- endpoint-error reduction: 45.63%;
- direction-error reduction: 87.91%;
- final total loss: 0.2477;
- final endpoint error: 12.90 pixels;
- final direction error: 8.66 degrees.

Formal training completed 30 epochs. It was resumed after epoch 2 only to
increase the batch size from 16 to 32; model and optimizer states were restored.
At the optimized batch size, steady-state epochs took approximately 8–12
seconds. The resume segment, including a 5 GiB float32 RPNet feature cache,
epochs 3–30 and final evaluation, took 570.3 seconds. Peak CUDA allocation was
3,036,230,144 bytes.

All fixed-threshold (`existence_threshold=0.5`) precision, recall and F1 values
were zero because no query probability crossed 0.5. This is reported as the
primary uncalibrated result and was not hidden by changing the training loss.
The behavior is consistent with unweighted BCE over six slots, where the
positive-slot fraction is small.

For diagnosis, an identical validation threshold sweep was applied to all
modalities. At the shared threshold 0.10:

| Modality | Precision | Recall | F1 | Endpoint mean | Direction mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.1210 | 0.6437 | 0.2037 | 16.00 px | 33.04 deg |
| no trajectory | 0.1092 | 0.5810 | 0.1839 | 16.33 px | 33.23 deg |
| trajectory + graph | 0.0920 | 0.3968 | 0.1494 | 19.10 px | 37.37 deg |

At this threshold, full improves F1 over no-trajectory by 0.0199. The gain is
not stable across the threshold sweep: no-trajectory reaches F1 0.2044 at
threshold 0.15, while full reaches F1 0.2037 at threshold 0.10. Therefore this
experiment does not establish a robust trajectory increment.

The calibrated predictions also expose query collapse. At threshold 0.10,
full has a duplicate-query pair ratio of 1.0 and 85.55% of validation nodes
contain duplicate predicted queries. This result is recorded as a Stage 3C
research failure mode; no diversity loss or later-stage anchor fusion was
introduced to mask it.

Because fixed-threshold F1 tied at zero, post-run checkpoint selection used the
documented F1-first, validation-loss tie-break. Epoch 16 had the minimum full
validation loss (0.723987) and is the selected checkpoint:

```text
/home/wangziyu/VecRoad_self/data_self/stage3c_branch_aux/checkpoints/stage3c_aux.best.pth.tar
SHA256 3f18e571031f673f79f5994971a3b16c2913327681ad3498c48b49b96d290b32
```

The latest epoch-30 checkpoint is retained separately. Auxiliary predictions
were never connected to `Path.push`.

Recorded figures:

- [formal training curves](stage3c_training_curves_20260724.png) (the plotted
  in-memory segment is epochs 3–30 after strict resume; the server-side JSONL
  retains all 30 epochs);
- [32-sample sanity curve](stage3c_sanity_curve_20260724.png).
