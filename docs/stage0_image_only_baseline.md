# road_self Stage 0 image-only baseline

## 1. Purpose and current status

Stage 0 establishes the image-only control for later trajectory research. In
`TRAJ.MODE: none`, road_self must not read trajectory data and must execute the
official VecRoad image network and graph-growth state machine.

The current implementation is stricter than the first Stage 0 revision:

- the registered image-only `RPNet` modules and state-dict contain the same
  648 keys as official VecRoad;
- the official `vecroad.pth.tar` loads with `strict=True`;
- feature fusion, walked-path input, segmentation supervision, recursive
  anchor feedback, and loss weights follow the official implementation;
- road_self configuration, checkpoint lifecycle, tests, diagnostics, and
  optional trajectory code are retained.

This does not mean the complete road_self repository is a byte-for-byte copy
of the official repository. Data handling, configuration, validation,
checkpoint metadata, and optional trajectory experiments remain fork-specific.

The pre-restoration full-resolution implementation is preserved by Git commit
`692bdaf` and tag `pre-original-architecture-20260723`. Its 1,191-key
image-only checkpoint is not compatible with the restored 648-key formal
baseline and must not be reported as an official-architecture result.

## 2. Invariants

The restoration does not change:

- `Path.pop` and `Path.push`;
- `map_to_coordinate`;
- `TargetPosesContainer` and target-map generation;
- `NUM_TARGETS=4` semantics;
- `STEP_LENGTH=20` semantics;
- graph serialization;
- trajectory loading/filtering algorithms used by `legacy_current`.

It also does not add any Stage 1 trajectory representation, branch query,
matching, projection, graph-state encoder, or new loss.

## 3. Image-only data and model contract

The active flow is:

1. `TRAJ.MODE: none` resolves to `use_traj=False`.
2. Dataset and inference construct `Path` with empty trajectory containers.
3. No trajectory file, raster, sequence, padding, normalization, or GPU
   tensor is requested.
4. A local aerial crop `[B,3,H,W]` and walked-path raster
   `[B,1,H/4,W/4]` enter `RPNet`.
5. Backbone side features are fused at 1/4 resolution.
6. Road and junction heads produce `[B,1,H/4,W/4]` logits.
7. The official recursive anchor decoder produces full-resolution
   `anchor` and `anchor_lowrs` logits `[B,NUM_TARGETS,H,W]`.
8. Between recursive targets, `avgpool4(decoded_ft_1)` is written back into
   the 1/4-resolution fusion tensor.
9. Inference continues through `map_to_coordinate` and `Path.push`.

The core output keys remain:

- `road`;
- `junc`;
- `anchor`;
- `anchor_lowrs`.

Compatibility outputs such as `traj_road` and `feature_maps` remain available.

## 4. Official loss and target resolution

Road and junction targets are the original 1/4-resolution maps. No label
dilation or full-resolution thick-5 segmentation target is used by the active
training loss. The full-resolution thickness-3 road raster remains available
only to the existing graph-following code.

```text
anchor_loss = anchor_final_loss + anchor_mid_loss
total_loss = anchor_loss + road_loss + junc_loss
```

All component losses retain the original summed binary-cross-entropy form.

## 5. Trajectory modes

`utils/trajectory_mode.py` is the centralized resolver.

- `none`: strict image-only path; trajectory modules are not registered in
  `RPNet` and the state-dict is official-checkpoint compatible.
- `legacy_current`: preserves the existing trajectory loader, fixed-circle
  filter, sequence Transformer, missing-trajectory placeholder, and optional
  DSF modules. These features now attach at the official 1/4-resolution
  fusion boundary.

If `TRAJ.MODE` is absent, old `TRAIN.USE_TRAJ=False/True` maps to
`none/legacy_current`. New configuration wins on conflict and emits a warning.
Unknown and reserved modes fail explicitly.

Because the base network resolution has changed, old trajectory-trained
checkpoints should be treated as migration inputs, not as validated performance
checkpoints. The legacy permissive loader remains available, but a new
trajectory baseline should be retrained before comparison.

## 6. Inference threshold defect

`ROAD_SEG_THRESHOLE` is a misleading legacy name: `infer_anchor` also passes
it to `map_to_coordinate` as the anchor heatmap threshold. At `0.01`, nearby
anchor responses merge into connected components exceeding
`JUNC_MAX_REGION_AREA=200`; those components are discarded, which caused the
observed near-empty graph.

The baseline, smoke, Xian, official default, and now `default_self.yml` use
`0.3`. The implementation and semantics of `map_to_coordinate` were not
changed. On an earlier real-window diagnosis, `0.01` produced components of
296/284/263/260 pixels and zero coordinates, while `0.3` produced four valid
coordinates.

## 7. Checkpoint lifecycle

The formal original-architecture baseline uses:

```yaml
TRAIN:
  CHECKPOINT:
    PREFIX: image_only_original
    SAVE_LATEST: True
    SAVE_EVERY_OUTER: 1
TEST:
  CKPT_FILE: image_only_original.latest.pth.tar
```

Training writes, for example:

- `data_self/baseline_image_only_original/ckpt/image_only_original.outer_001.path_2048.pth.tar`;
- `data_self/baseline_image_only_original/ckpt/image_only_original.latest.pth.tar`.

`infer.py` resolves the same latest path and loads `TEST.CKPT_FILE` strictly.
Legacy `TEST.CKPT: vecroad2` still resolves to
`DIR.CHECK_POINT_DIR/vecroad2.pth.tar` and uses the historical permissive
loader.

Checkpoint payloads include model/optimizer state, outer/path iteration,
trajectory mode, configuration snapshot/path, seed, model name,
`NUM_TARGETS`, `STEP_LENGTH`, and `WINDOW_SIZE`.

The two-batch smoke configuration writes:

- `data_self/baseline_image_only_smoke/ckpt/image_only_original_smoke.outer_001.path_0002.pth.tar`;
- `data_self/baseline_image_only_smoke/ckpt/image_only_original_smoke.latest.pth.tar`.

## 8. Validation commands

Run unit tests:

```bash
python -m unittest discover -s tests -v
```

Validate configuration, trajectory bypass, and synthetic forward equivalence:

```bash
python scripts/validate_stage0_baseline.py \
  --device cpu --input-size 64 --batch-size 1 \
  --seed 20260722 --tolerance 1e-6
```

Validate the production forward against an independent transcription of the
official VecRoad forward:

```bash
python scripts/validate_original_vecroad_alignment.py \
  --checkpoint /path/to/official/vecroad.pth.tar \
  --device cuda --input-size 256 --batch-size 1 --tolerance 1e-6
```

Run real two-batch train/save/load/infer preparation:

```bash
python scripts/smoke_stage0_training.py \
  --config configs/baseline_image_only_smoke.yml
```

Run a bounded real-data closed loop with an explicit compatible checkpoint:

```bash
python scripts/validate_stage0_closed_loop.py \
  --config configs/baseline_image_only.yml \
  --checkpoint /path/to/official/vecroad.pth.tar \
  --region xian --start-x 1704 --start-y 794 \
  --start-state key_point --max-iterations 100 \
  --seed 20260722 --device cuda --tolerance 1e-6 \
  --coordinate-tolerance 1e-6 \
  --output-dir data_self/baseline_image_only/closed_loop/original_official_100
```

Formal training and inference use:

```bash
python train.py --config configs/baseline_image_only.yml
python infer.py --config configs/baseline_image_only.yml
```

The second command requires the formal latest checkpoint created by the first.
The formal run is isolated under `data_self/baseline_image_only_original` so
that TensorBoard and checkpoints cannot be mixed with the archived
full-resolution Stage 0 experiment. `TRAIN.DETECT_ANOMALY` defaults to `False`
for the formal run and can be enabled explicitly for debugging.

## 9. Executed results on 2026-07-23

Local environment:

- PyTorch `2.8.0+cu126`;
- unit tests: 36/36 passed;
- Stage 0 synthetic validation: passed;
- production-vs-official-reference synthetic forward: all four outputs finite,
  maximum and mean absolute differences `0.0`;
- image-only state-dict: 648 keys and no trajectory parameters.

237 server environment:

- Python `3.8.18`, PyTorch `2.4.1+cu121`, RTX 4090;
- official checkpoint `/home/wangziyu/VecRoad-master/data/ckpt/vecroad.pth.tar`
  loaded strictly with 648/648 keys;
- official-checkpoint production-vs-reference comparison at 256×256:
  `road`, `junc`, `anchor`, and `anchor_lowrs` all had
  `max_abs_diff=0.0`, `mean_abs_diff=0.0`, and finite values;
- two real Xian training batches completed forward, all four losses, backward,
  optimizer step, `push_and_vis_batch`, versioned/latest save, strict inference
  reload, and reload forward validation;
- smoke latest checkpoint has 648 model keys, 363 optimizer-state entries,
  `outer_it=1`, `path_it=1`, `trajectory_mode=none`, seed `20260722`, and the
  expected model constants;
- the forbidden trajectory probe path was not created;
- smoke lifecycle elapsed about 37.1 seconds.

The 100-step bounded Xian request used the official checkpoint and start
`(1704,794)`, selected as a degree-5 GT junction. The path naturally exhausted
after 94 model iterations. Results:

- 302 vertices;
- 604 directed edges;
- 302 undirected edges;
- per-step four-output difference between legacy-disabled and `none`: `0.0`;
- per-step `map_to_coordinate` and graph counts matched;
- final canonical and ID-based graph signatures matched;
- no trajectory path was created or read.

Detailed server report:

`data_self/baseline_image_only/closed_loop/original_official_100/report.json`

## 10. Evaluation commands and unclaimed results

After full inference, existing evaluation entry points remain:

```bash
python eval/graph2wkt.py --graph_dir <graph_dir> --save_dir <wkt_dir>
python eval/eval_apls_metric.py --wkt_dir <wkt_dir> --gt_dir <gt_wkt_dir> \
  --save_dir <result_dir> --apls_path <visualizer.jar>
python eval/eval_junction_metric.py --graph_dir <graph_dir> \
  --gt_dir <gt_graph_dir> --save_dir <result_dir> --file_name <name>
python eval/graph2seg.py --graph_dir <graph_dir> --save_dir <seg_dir> \
  --region_file <regions.txt> --img_size 4096 --thickness 5
python eval/eval_pixel_metric.py --gt_dir <gt_mask_dir> \
  --pred_dir <seg_dir> --thresh 128 --relax 6
python metrics/topo_metric.py
```

No APLS, TOPO, Junction-F1, pixel score, full-city inference score, or
multi-seed result is claimed here. The smoke checkpoint is not a performance
checkpoint. The old 1,191-key full-resolution checkpoint is historical and is
not an original-architecture baseline.

## 11. Remaining Stage 0 experiment work

The code-path contract, official checkpoint compatibility, real two-batch
training lifecycle, and non-empty bounded closed loop pass. Before publishing
an image-only road_self score, a fresh formal checkpoint should be trained with
`configs/baseline_image_only.yml`, followed by full inference and the existing
metrics over the intended AOI and multiple seeds.

This is experiment completion work, not a missing code-path repair. Stage 1
should use the 648-key image-only path as the no-trajectory fallback reference.
