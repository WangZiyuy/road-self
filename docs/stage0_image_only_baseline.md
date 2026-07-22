# road_self Stage 0/0.5 image-only baseline

## 1. Purpose and scope

This is the image-only baseline of the `road_self` fork. It provides a stable
reference that later trajectory work must recover when trajectory input is
unavailable. It is not a claim of strict numerical reproduction of the
official VecRoad repository: the baseline freezes the current `road_self`
image branch and verifies that the new configuration/call path is equivalent
to this fork's legacy `USE_TRAJ=False` call path.

The formal Xian run described below used repository revision `31183e8` plus
the documented Xian data-path configuration.
It does not change `Path.pop`, `Path.push`, `map_to_coordinate`,
`TargetPosesContainer`, target-map generation, `NUM_TARGETS`, `STEP_LENGTH`,
the recursive anchor head, anchor feedback, loss weights, graph serialization,
or the `legacy_current` trajectory algorithm.

## 2. Stage 0 audit and data path

Before Stage 0, `TRAIN.USE_TRAJ=False` selected the original `fuse_module`, but
the surrounding data and training code still loaded, filtered, padded,
normalized, and transferred trajectory inputs. Stage 0 centralized the mode
resolution in `utils/trajectory_mode.py` and made `TRAJ.MODE: none` bypass
those dependencies.

The image-only data flow is now:

1. `OSMDataset` and inference construct `Path` with empty trajectory state.
2. Local input construction requests the aerial image, walked path, and the
   original training targets only.
3. Training and inference pass `None` for every trajectory tensor and
   `use_traj=False`.
4. `RPNet` uses the image backbone, road/junction heads, original
   `fuse_module`, and the unchanged recursive anchor feedback.
5. Inference continues through the original `map_to_coordinate` and
   `Path.push` state machine.

The required model outputs remain `road`, `junc`, `anchor`, and
`anchor_lowrs`. The existing additional compatibility outputs may still be
present.

## 3. Trajectory modes

`utils/trajectory_mode.py` is the only trajectory-mode resolver.

- `none`: no external trajectory load, fetch, rasterization, filtering,
  padding, normalization, device transfer, or fusion.
- `legacy_current`: the existing `road_self` trajectory implementation,
  preserved as an ablation baseline.
- `structured_all` and `branch_slot`: reserved and deliberately unsupported in
  Stage 0/0.5.

`TRAJ.MODE` takes precedence. If it is absent, legacy
`TRAIN.USE_TRAJ=False/True` maps to `none/legacy_current`. A conflicting new
and old setting emits a warning; an unknown or reserved mode raises
`ValueError`. The historical experiment configurations were not migrated, so
their legacy behavior is retained.

## 4. Stage 0.5 additions

### 4.1 Image-only visualization safety

`Path` initializes both `valid_trajectories` and `circles` as empty lists.
`visualize_output` draws trajectory points and the fixed-radius legacy marker
only when the corresponding legacy state exists. An image-only path therefore
does not access an uninitialized `circles` attribute and does not draw the
radius-50 trajectory-filter circle. `SAVE_EXAMPLES=True` is covered through
the real `OSMDataset.push_and_vis_batch -> Path.visualize_output` call path.

The formal baseline sets `TRAIN.SAVE_EXAMPLES: False` to avoid unnecessary
training I/O. Enabling it remains supported.

### 4.2 Configurable path iterations

`TRAIN.PATH_ITERATIONS` controls the number of training path-expansion
iterations per outer iteration. If absent, it defaults to `2048`, preserving
old configurations. The loop bound, log denominator, TensorBoard global step,
and checkpoint `path_it` now use the resolved value. This setting is separate
from `MAX_PATH_LENGTH`.

### 4.3 Checkpoint lifecycle

The new lifecycle is centralized in `utils/checkpoint_utils.py`. For the
formal baseline:

```yaml
TRAIN:
  CHECKPOINT:
    PREFIX: image_only
    SAVE_LATEST: True
    SAVE_EVERY_OUTER: 1
TEST:
  CKPT_FILE: image_only.latest.pth.tar
```

At the end of an eligible outer iteration, training writes:

- versioned: `data_self/baseline_image_only/ckpt/image_only.outer_001.path_2048.pth.tar`;
- latest: `data_self/baseline_image_only/ckpt/image_only.latest.pth.tar`.

The version numbers vary with the current outer/path iteration. `infer.py`
resolves the same latest path from `TEST.CKPT_FILE`, checks that it exists, and
loads it strictly. No copy or rename is required.

New checkpoints include:

- model `state_dict` without renamed parameter keys;
- optimizer state;
- zero-based `outer_it` and `path_it` metadata;
- trajectory mode;
- configuration path and serializable configuration snapshot;
- random seed, model name, `NUM_TARGETS`, `STEP_LENGTH`, and `WINDOW_SIZE`.

`TEST.CKPT_FILE` has priority over legacy `TEST.CKPT`. Relative paths are
resolved below `DIR.CHECK_POINT_DIR`; absolute paths remain absolute. With only
`TEST.CKPT: vecroad2`, inference still resolves
`DIR.CHECK_POINT_DIR/vecroad2.pth.tar` and uses the historical permissive
loader. A conflicting new/old path emits a warning. A missing file error
contains the fully resolved path.

### 4.4 Smoke configuration and runner

`configs/baseline_image_only_smoke.yml` keeps the baseline model/loss behavior
but uses one outer iteration, two path iterations, batch size one, no example
rendering, no pretrained-backbone download, and no TensorBoard dependency. It
writes:

- `data_self/baseline_image_only_smoke/ckpt/image_only_smoke.outer_001.path_0002.pth.tar`;
- `data_self/baseline_image_only_smoke/ckpt/image_only_smoke.latest.pth.tar`.

`scripts/smoke_stage0_training.py` runs training, points the trajectory
directory at a forbidden nonexistent path, verifies that the path was not
created, invokes `infer.prepare_net()` against the saved latest checkpoint,
and runs checkpoint-based forward equivalence. Optional graph/region/tile
overrides support installations whose data layout differs from the config.

### 4.5 Validation and canonical graph comparison

`scripts/validate_stage0_baseline.py` retains configuration, no-trajectory,
and synthetic-forward validation. With `--checkpoint`, it prints metadata and
state-dict key count, checks parameter names and shapes, strictly loads the
same state dict, and compares the legacy `use_traj=False` call with the new
all-`None` trajectory call.

Graph comparison is ID/order independent. Its canonical signature contains:

- the vertex-coordinate multiset;
- directed coordinate endpoint pairs;
- normalized undirected coordinate endpoint pairs;
- vertex, directed-edge, and undirected-edge counts.

Coordinates are quantized by `--coordinate-tolerance` (default `1e-6`). The
old ID-based signature is retained as diagnostic output, while the canonical
geometry/topology signature determines equivalence.

### 4.6 Bounded real-data closed-loop validation

`scripts/validate_stage0_closed_loop.py` keeps two independent `Path` states
on one fixed real aerial tile and start point. The legacy-disabled model call
supplies trajectory-shaped tensors with `use_traj=False`; the Stage-0 call
supplies `None`. Neither path reads a trajectory file. At every iteration the
script compares all four core logits, `map_to_coordinate` output, and graph
counts after `Path.push`, then saves both graphs for canonical comparison.

The first formal check exposed that the new baseline configs had copied the
modified `default_self.yml` inference threshold `0.01`. At this threshold the
anchor response around a junction merged into regions larger than
`JUNC_MAX_REGION_AREA`, so no coordinate survived. Official VecRoad's
`configs/default.yml`, this repository's `configs/default.yml`, and
`configs/xian_self.yml` use `ROAD_SEG_THRESHOLE: 0.3`. The two new image-only
baseline configs now use `0.3`; historical configs and `legacy_current` remain
unchanged. No model weight, loss, `map_to_coordinate` implementation, or graph
state transition was changed. The checkpoint's embedded training snapshot
still records the pre-correction TEST value; this does not affect its trained
weights because the field is inference-only.

## 5. Model and loss contract

Image-only inputs are an aerial image `[B,3,H,W]`, walked path `[B,1,H,W]`,
`model="origin"`, `use_traj=False`, and `None` for trajectory tensors/masks.
Output shapes are:

- `road`: `[B,1,H,W]`;
- `junc`: `[B,1,H,W]`;
- `anchor`: `[B,NUM_TARGETS,H,W]`;
- `anchor_lowrs`: `[B,NUM_TARGETS,H,W]`.

The loss remains:

```text
anchor_loss = anchor_final_loss + anchor_mid_loss
total_loss = anchor_loss + 10 * road_loss + 10 * junc_loss
```

No trajectory, branch, reliability, or other new loss was added.

## 6. Commands

Run all tests:

```bash
python -m unittest discover -s tests -v
```

Run deterministic synthetic validation:

```bash
python scripts/validate_stage0_baseline.py \
  --device auto --input-size 64 --batch-size 1 \
  --seed 20260722 --tolerance 1e-6
```

Validate a compatible checkpoint:

```bash
python scripts/validate_stage0_baseline.py \
  --device auto --input-size 64 --batch-size 1 \
  --seed 20260722 --tolerance 1e-6 \
  --checkpoint data_self/baseline_image_only/ckpt/image_only.latest.pth.tar
```

Run the bounded real-data closed-loop comparison:

```bash
python scripts/validate_stage0_closed_loop.py \
  --config configs/baseline_image_only.yml \
  --checkpoint data_self/baseline_image_only/ckpt/image_only.latest.pth.tar \
  --region xian --start-x 1704 --start-y 794 \
  --start-state key_point --max-iterations 20 \
  --seed 20260722 --device cuda --tolerance 1e-6 \
  --coordinate-tolerance 1e-6 \
  --output-dir data_self/baseline_image_only/closed_loop/formal_latest_threshold03
```

Compare independently generated legacy/new closed-loop graph files:

```bash
python scripts/validate_stage0_baseline.py \
  --device cpu --input-size 64 --batch-size 1 \
  --legacy-graph outputs/legacy_use_traj_false.graph \
  --stage0-graph outputs/traj_mode_none.graph \
  --coordinate-tolerance 1e-6
```

Train and infer the formal baseline:

```bash
python train.py --config configs/baseline_image_only.yml
python infer.py --config configs/baseline_image_only.yml
```

Run a two-batch smoke on a complete installation:

```bash
python scripts/smoke_stage0_training.py \
  --config configs/baseline_image_only_smoke.yml
```

For an engineering-only integration check when the source GT training graph is
unavailable, the committed small graph fixture can be used with installed
imagery:

```bash
python scripts/smoke_stage0_training.py \
  --config configs/baseline_image_only_smoke.yml \
  --graph-dir tests/fixtures/stage0_smoke \
  --region-path data_self/input/regions/xian_regions.txt \
  --tile-dir data_self/input/imagery
```

This fixture command checks the code lifecycle only; it is not a training or
quality result.

## 7. Evaluation commands

Assuming inference produced graphs under
`data_self/baseline_image_only/graphs/image_only.latest_4/post`:

```bash
python eval/graph2wkt.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only.latest_4/post \
  --save_dir data_self/baseline_image_only/graphs/image_only.latest_4/post_wkt

python eval/eval_apls_metric.py \
  --file_name image_only_post_apls \
  --wkt_dir data_self/baseline_image_only/graphs/image_only.latest_4/post_wkt \
  --gt_dir data_self/input/graphs_test_wkt \
  --save_dir data_self/baseline_image_only/graphs/image_only.latest_4 \
  --apls_path eval/apls-visualizer-1.0/visualizer.jar

python eval/eval_junction_metric.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only.latest_4/post \
  --gt_dir data_self/input/graphs \
  --save_dir data_self/baseline_image_only/graphs/image_only.latest_4 \
  --file_name image_only_post_jf1

python eval/graph2seg.py \
  --graph_dir data_self/baseline_image_only/graphs/image_only.latest_4/post \
  --save_dir data_self/baseline_image_only/graphs/image_only.latest_4/post_seg \
  --region_file data_self/input/regions/test_regions.txt \
  --img_size 4096 --thickness 5

python eval/eval_pixel_metric.py \
  --gt_dir data_self/input/mask_test \
  --pred_dir data_self/baseline_image_only/graphs/image_only.latest_4/post_seg \
  --thresh 128 --relax 6
```

The existing TOPO implementation contains dataset-specific paths and an MBR.
After configuring those existing inputs, its current entry point is:

```bash
python metrics/topo_metric.py
```

Stage 0.5 did not change evaluation code.

## 8. Executed verification

Local executable checks used PyTorch `2.8.0+cu126` and an RTX 4050 Laptop GPU.
Server checks used Python `3.8.18`, PyTorch `2.4.1+cu121`, and an RTX 4090.

- Local `unittest`: 33/33 passed.
- Server `unittest`: 33/33 passed.
- Local and server synthetic validation at `64x64`: all configuration and
  no-trajectory dependency checks passed; all four output tensors were finite;
  maximum and mean differences were `0.0`.
- Tiny-model checkpoint unit test: model and Adam optimizer state restored;
  output maximum difference was `0.0`.
- Server two-batch integration smoke with the supplied Xian graph and aerial
  imagery: all four 2048x2048 training subtiles were accepted; two
  forward/loss/backward/optimizer steps and `push_and_vis_batch` completed;
  losses remained finite; both checkpoint files were written; elapsed
  lifecycle time was about 35.8 seconds.
- The nonexistent trajectory probe path was not created.
- `infer.prepare_net()` directly loaded the generated latest checkpoint.
- The checkpoint contained 1,191 model keys and metadata
  `outer_it=1`, `path_it=1`, `trajectory_mode=none`, `model_name=origin`,
  `NUM_TARGETS=4`, `STEP_LENGTH=20`, `WINDOW_SIZE=256`, and seed `20260722`.
- Checkpoint-based legacy/new forward comparison passed for `road`, `junc`,
  `anchor`, and `anchor_lowrs`; all maximum and mean differences were `0.0`
  at tolerance `1e-6`, and all outputs were finite.
- Canonical graph tests proved equality across different vertex/edge IDs and
  insertion order, and detected a changed edge.
- A formal image-only Xian training run used `TRAJ.MODE=none`, seed `20260722`,
  batch size 2, a 4096x4096 training tile, and an RTX 4090. It ran for about
  3 hours 51 minutes. Seven complete outer iterations (14,336 optimizer
  steps) were saved; training was intentionally stopped during outer 8 at
  path iteration 757. The incomplete outer iteration was not checkpointed.
- The frozen formal checkpoint is
  `data_self/baseline_image_only/ckpt/image_only.outer_007.path_2048.pth.tar`;
  `image_only.latest.pth.tar` resolves to the same completed state. It contains
  1,191 model keys, 365 optimizer-state entries, one optimizer parameter group,
  and metadata `outer_it=7`, `path_it=2047`, `trajectory_mode=none`,
  `model_name=origin`, `NUM_TARGETS=4`, `STEP_LENGTH=20`, `WINDOW_SIZE=256`,
  and seed `20260722`. The frozen versioned file has SHA-256
  `dce838697a03fc5dfa9a27f35a920faddfcd9c2c2d4fbf0603926ed90b8f5c3b`.
- Checkpoint-based CUDA comparison of legacy `use_traj=False` and
  `TRAJ.MODE=none` passed for `road`, `junc`, `anchor`, and `anchor_lowrs`.
  Every maximum and mean absolute difference was `0.0` at tolerance `1e-6`;
  every tensor was finite.
- The bounded Xian closed loop used fixed start `(1704, 794)`, requested 20
  iterations, and completed all 20. Every per-step core-output difference was
  `0.0`; `map_to_coordinate` and post-`Path.push` graph counts matched at every
  step. The two final graphs each contain 72 vertices, 142 directed edges, and
  71 undirected edges. Both canonical and ID-based signatures are equal.
- The closed-loop trajectory probe path was neither accessed nor created.
- The combined machine-readable report is
  `data_self/baseline_image_only/run/final_stage0_validation.json`; the detailed
  closed-loop trace is
  `data_self/baseline_image_only/closed_loop/formal_latest_threshold03/report.json`.

The supplied Xian graph contains 762 vertices and 1,730 directed edges (865
bidirectional road segments), with no invalid references, duplicate directed
edges, self-edges, isolated vertices, or missing reverse edges. Its coordinates
match the metadata's 4300x5000 source-image extent. The current training loader
uses only the first 4096x4096 tile; consequently 168 vertices and 388 directed
edges outside that tile are not sampled by the current baseline training path.
This does not block the smoke lifecycle, but full-AOI coverage should be
resolved before treating a future full-AOI run as a published baseline.

## 9. Not executed and not claimed

- Full unbounded `infer.py` exploration over all predicted Xian starting
  points was not run. The executed result is a fixed-start, 20-step closed-loop
  equivalence check, not a complete extracted city graph.
- APLS, TOPO, Junction-F1, and pixel metrics were not run. No value for these
  metrics is claimed.
- The formal run covers one 4096x4096 crop and one random seed. It is not a
  full-AOI or multi-seed paper result.
- A trajectory-trained checkpoint is not treated as image-only performance.
  Such a compatible checkpoint may only be used to test equivalence of the two
  calls while trajectory use is disabled.

## 10. Remaining Stage 0 baseline work

The Stage 0/0.5 engineering lifecycle is covered: real training, latest and
versioned checkpoint save/load, optimizer metadata, no-trajectory gating,
checkpoint forward equivalence, and a non-empty bounded closed-loop graph pair
all pass. The checkpoint can therefore be frozen as the current engineering
reference for later no-trajectory fallback tests.

Before claiming a published image-only baseline score, the remaining
experiment work is to resolve full 4300x5000 AOI coverage, run full inference,
record APLS/TOPO/Junction-F1/pixel metrics, and repeat the selected experiment
with multiple seeds. These are performance-completeness tasks, not blockers
for the Stage 0 code-path contract.

Legacy fixed circles, hard filtering, trajectory Transformer behavior, and
trajectory cache structure remain intentionally unchanged and do not execute
in `TRAJ.MODE=none`.
