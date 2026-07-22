# Stage 0: image-only VecRoad baseline

## 1. Goal and frozen scope

Stage 0 establishes a reproducible image-only baseline that later trajectory
work must be able to recover when trajectory input is unavailable. It does not
change `Path.pop`, `Path.push`, `map_to_coordinate`, target-map generation,
`NUM_TARGETS`, `STEP_LENGTH`, the recursive anchor head, or graph serialization.

Baseline source revision: `87ded93f70a110b10672b384b6c0ebc04ba5dc22`.
The working repository remote is `WangZiyuy/vecroad-self`, although the task
description referred to `WangZiyuy/road-self`.

## 2. Pre-change audit

Before this stage, `TRAIN.USE_TRAJ=False` selected the original
`fuse_module` in `RPNet.forward`, so trajectory Transformer features and
`missing_traj_feature` were not used by the anchor head. It did not, however,
create an image-only data path:

- `OSMDataset.__init__` always passed a trajectory directory to `Tiles`, loaded
  every region's prepared/raw trajectories, and passed them to `Path`.
- `OSMDataset.get_batch` always requested trajectory raster fields and
  `valid_trajectories`. Both values of `TRAIN.TRAJ_FILTER` called the same
  `filter_trajectories_on_gpu2` implementation.
- `train.py` always copied trajectory rasters to CUDA and always padded and
  normalized trajectory sequences, even when `USE_TRAJ=False`.
- `infer.py` partially gated trajectory sequence loading with `USE_TRAJ`, but
  trajectory configuration was still distributed through the inference code.
- The model's image-only outputs were `road`, `junc`, `anchor`, and
  `anchor_lowrs`; the fork also returned `traj_road` and `feature_maps` for
  compatibility.
- The graph exploration loop remained
  `Path.pop -> make_path_input -> model -> map_to_coordinate -> Path.push` and
  did not require real trajectory objects when empty containers were supplied.

The main baseline risk was therefore hidden trajectory file, preprocessing,
CUDA-memory, and failure dependencies despite an image-only model branch.

## 3. Current image-only data flow

With `TRAJ.MODE: none`, the active path is:

1. `OSMDataset`/inference construct `Path` with `None`, an empty trajectory
   list, and no trajectory index.
2. Local input construction requests only the aerial image, walked path, and
   the original supervision fields needed during training.
3. Training/inference pass `None` for every trajectory tensor and set
   `use_traj=False`.
4. `RPNet` uses the original image backbone, road/junction heads, original
   `fuse_module`, recursive anchor feedback, and unchanged output schema.
5. Inference applies the unchanged `map_to_coordinate` and `Path.push` logic.

No trajectory directory is required by
`configs/baseline_image_only.yml`.

## 4. Trajectory modes and compatibility

`utils/trajectory_mode.py` is the single mode resolver.

- `none`: do not load, fetch, rasterize, filter, pad, normalize, transfer, or
  fuse external trajectories.
- `legacy_current`: preserve the repository's existing trajectory
  implementation as a future ablation baseline.
- `structured_all` and `branch_slot`: reserved names only; they raise
  `ValueError` in stage 0.

Resolution order:

1. If `TRAJ.MODE` exists, it wins.
2. Otherwise, legacy `TRAIN.USE_TRAJ=False` maps to `none`, and `True` maps to
   `legacy_current`.
3. A conflicting new and legacy setting emits a `RuntimeWarning`; the new
   setting wins.
4. An unknown mode raises `ValueError`.

Because `DSFNet` consumes a trajectory raster, `none` also requires
`TRAIN.MODEL: origin`; an incompatible explicit combination raises a clear
`ValueError` instead of reaching a `None` tensor failure inside the model.

The historical `configs/default_self.yml` was not modified, so its
`TRAIN.USE_TRAJ=True` setting resolves to `legacy_current` and retains the old
experiment behavior.

## 5. Modified files

- `utils/trajectory_mode.py`: centralized mode resolution and testable gates
  for region loading, fetch fields, and sequence preprocessing.
- `configs/baseline_image_only.yml`: complete image-only training/inference
  configuration with `MODEL: origin`, `NUM_TARGETS: 4`, and unchanged
  VecRoad window/step settings.
- `utils/OSMDataset.py`: skips every trajectory allocation, load, fetch, and
  filter operation in `none` mode; trajectory keys are absent from returned
  batches.
- `train.py`: passes `None` trajectory arguments and avoids all trajectory CUDA
  and preprocessing work in `none` mode.
- `infer.py`: avoids region trajectory loading and trajectory fetch/preparation
  in `none` mode while retaining the original graph-growth state machine.
- `model/model.py`: adds an optional `backbone_pretrained=False` construction
  switch solely for offline tests; the production default remains `True` and
  the state-dict schema is unchanged.
- `scripts/validate_stage0_baseline.py`: configuration, dependency-gating,
  forward-equivalence, and optional closed-loop graph comparison.
- `tests/test_trajectory_mode.py` and `tests/test_image_only_path.py`: lightweight
  `unittest` coverage without adding a new test dependency.

## 6. Model contract and loss

Image-only model inputs are:

- aerial image: `[B, 3, H, W]`;
- walked path: `[B, 1, H, W]`;
- trajectory image, aerial/trajectory concatenation, trajectory sequence, and
  valid mask: all `None`;
- `model="origin"`, `use_traj=False`.

Required outputs remain:

- `road`: `[B, 1, H, W]`;
- `junc`: `[B, 1, H, W]`;
- `anchor`: `[B, NUM_TARGETS, H, W]`;
- `anchor_lowrs`: `[B, NUM_TARGETS, H, W]`.

Training still computes summed binary cross entropy. The existing code first
adds the recursive low-resolution anchor loss into `anchor_loss`, then uses:

```text
anchor_loss = anchor_final_loss + anchor_mid_loss
total_loss = anchor_loss + 10 * road_loss + 10 * junc_loss
```

No trajectory, branch, or reliability loss was added.

## 7. Reproduction commands

Run all stage-0 tests:

```bash
python -m unittest discover -s tests -v
```

Run deterministic synthetic forward validation:

```bash
python scripts/validate_stage0_baseline.py \
  --device auto --input-size 64 --batch-size 1 \
  --seed 20260722 --tolerance 1e-6
```

An optional compatible checkpoint can be supplied with `--checkpoint`. For a
closed-loop comparison, first produce two graph files from identical seeds and
starting state, then add `--legacy-graph OLD.graph --stage0-graph NEW.graph`.

Train the baseline:

```bash
python train.py --config configs/baseline_image_only.yml
```

For inference, place or name a compatible image-only checkpoint as
`data_self/baseline_image_only/ckpt/image_only.pth.tar` (or change only
`TEST.CKPT` in the configuration), then run:

```bash
python infer.py --config configs/baseline_image_only.yml
```

## 8. Evaluation commands

Assuming inference produced graphs under
`data_self/baseline_image_only/graphs/image_only_4/post`:

```bash
python eval/graph2wkt.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only_4/post \
  --save_dir data_self/baseline_image_only/graphs/image_only_4/post_wkt

python eval/eval_apls_metric.py \
  --file_name image_only_post_apls \
  --wkt_dir data_self/baseline_image_only/graphs/image_only_4/post_wkt \
  --gt_dir data_self/input/graphs_test_wkt \
  --save_dir data_self/baseline_image_only/graphs/image_only_4 \
  --apls_path eval/apls-visualizer-1.0/visualizer.jar

python eval/eval_junction_metric.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only_4/post \
  --gt_dir data_self/input/graphs \
  --save_dir data_self/baseline_image_only/graphs/image_only_4 \
  --file_name image_only_post_jf1

python eval/graph2seg.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only_4/post \
  --save_dir data_self/baseline_image_only/graphs/image_only_4/post_seg \
  --region_file data_self/input/regions/test_regions.txt \
  --img_size 4096 --thickness 5

python eval/eval_pixel_metric.py \
  --gt_dir data_self/input/mask_test \
  --pred_dir data_self/baseline_image_only/graphs/image_only_4/post_seg \
  --thresh 128 --relax 6
```

The repository's current TOPO implementation has no command-line arguments and
contains dataset-specific absolute paths and an MBR in `metrics/topo_metric.py`.
After configuring those existing evaluation inputs for the target region, its
current entry point is:

```bash
python metrics/topo_metric.py
```

This stage intentionally does not alter evaluation code or its dataset-specific
assumptions.

## 9. Executed verification

Environment used for the executable checks:

- PyTorch `2.8.0+cu126`;
- CUDA, one NVIDIA RTX 4050 Laptop GPU;
- synthetic input seed `20260722`;
- one model instance in `eval` mode and one shared state dict;
- no checkpoint loaded (`synthetic_initialization`).

Results:

- `python -m unittest discover -s tests -v`: 13/13 tests passed.
- Validation script at `64 x 64`: configuration and no-dependency checks
  passed; forbidden trajectory loader/padding/normalization call counts were
  all zero.
- `road`, `junc`, `anchor`, and `anchor_lowrs` were finite and had both maximum
  and mean absolute difference `0.0` between the legacy `use_traj=False` call
  and the new `TRAJ.MODE=none` call. The acceptance threshold was `1e-6`.

## 10. Not executed

- Full training was not run because it is a long experiment and the available
  Python environment lacks some repository runtime packages (`easydict` and
  TensorBoard). Tests use a local test-only EasyDict substitute.
- Checkpoint-based inference and metric evaluation were not run because the
  baseline checkpoint path does not exist. The available checkpoints are large
  trajectory-trained experiment checkpoints and are not treated as a valid
  image-only baseline.
- Closed-loop node/edge equality was not executed for the same checkpoint
  reason. The validation script retains a graph-pair comparison interface.
- No APLS, TOPO, Junction-F1, or pixel score is claimed in this stage report.

## 11. Deferred legacy trajectory issues

The following existing issues remain intentionally untouched for later stages:

- fixed circles, thresholds, hard truncation, and hard trajectory filtering;
- identical behavior in both branches of `TRAIN.TRAJ_FILTER`;
- flattening trajectory identity/time structure in the current Transformer
  path;
- batch-level rather than per-sample missing-trajectory handling;
- legacy raster and structured trajectory paths living side by side;
- trajectory cache identity, ordering, original indices, and time attributes.

These do not execute in `TRAJ.MODE=none`.
