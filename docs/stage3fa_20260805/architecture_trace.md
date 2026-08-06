# Stage 3F-A anchor architecture trace

## Frozen RPNet heads

For the official-aligned `origin` path and a 256x256 local input:

| head | frozen input | input shape | frozen output |
|---|---|---:|---:|
| road | `road_fts = road_seg(stage_fuse)` | `[B,64,64,64]` | `conv_road_final`, `[B,1,64,64]` |
| junction | `junc_fts = junc_seg(stage_fuse)` | `[B,64,64,64]` | `conv_junc_final`, `[B,1,64,64]` |
| anchor low resolution | `next_step` | `[B,32,64,64]` | `next_step_final` then original bilinear x4, `[B,1,256,256]` per step |
| anchor full resolution | `decoded_ft_1` | `[B,32,256,256]` | `conv_final`, `[B,1,256,256]` per step |

`stage_fuse` is `[B,128,64,64]`. Road and junction heads share it, but
neither new Stage 3F-A residual is injected there. The anchor loop has four
recursive steps under the unchanged `NUM_TARGETS=4` semantics.

`next_step` is an ancestor of both anchor outputs, but it is not the last
common anchor-only feature: the full-resolution head passes it through four
decoder blocks. More importantly, `decoded_ft_1` is pooled into the recursive
anchor feedback for later steps. Injecting evidence into `next_step` or before
that feedback would change the frozen recursive state. Stage 3F-A therefore
uses one shared trajectory projection and gate, followed by two separate
zero-initialized adapters at the exact final pre-head features. The full-head
adapter is applied after the original feedback tensor has already been
computed. This keeps recursive feedback, road and junction outputs unchanged.

## Original supervision and decoding

- Targets come directly from `Path.get_target_poses` and
  `Path.generate_target_maps`; the latter uses the existing sigma-3 Gaussian,
  including the original multi-endpoint sum-and-clamp behavior.
- Key-point supervision uses channel 0. Other states retain the existing
  `TargetPosesContainer.get_supervision_end_index()` behavior.
- Training uses the official VecRoad
  `binary_cross_entropy_with_logits(prediction, target, reduction="sum")`
  spatial/recursive-map loss for both the full-resolution anchor and
  `anchor_lowrs`; per-sample sums are averaged across the cache batch. No new
  heatmap, Dice, branch, support, or reliability loss is introduced.
- A failed preflight initially reused road_self's historical `BCEDiceLoss`
  with the unchanged `(prediction, target)` call order. That class expects the
  reverse order, so unconstrained low-resolution logits became BCE targets,
  the loss turned negative, and anchor AP collapsed. Those checkpoints are
  quarantined as failed diagnostics and are not Stage 3F-A results.
- Immediate-node metrics call the existing `map_to_coordinate` on channel 0,
  with the same threshold, connected-component behavior, area rejection and
  coordinate convention. All ablations use the same threshold and NMS path.

## Checkpoint boundary

The image RPNet, E4 trajectory/graph/branch modules, Stage 3E-3 M=1 evidence
encoder, and Stage 3D-A support head are loaded strictly and frozen. Only the
new trajectory projection, two residual adapters, and shared sample gate are
optimized. The canonical evidence checkpoint is the preselected seed
20260724 checkpoint; its required SHA256 is
`d7f00a12a10c8e80687945f0f68596883bdd4b038893d828bba297601b451ff6`.

The final 3x3 convolution in each residual adapter has zero weights and zero
bias. Thus both a present-trajectory initialization and every no-trajectory
call recover the cached original anchor logits exactly.

Both cached pre-head feature tensors use `float32`. The recursive low-resolution
state can exceed the finite `float16` range, while `float16` quantization of the
otherwise finite full-resolution feature changed reconstructed anchor logits by
about `1.01e-2` in a preflight sample. The final cache builder therefore rejects
NaN/Inf and reconstructs both anchor heads from the post-conversion arrays; the
full-resolution loss, low-resolution loss, and their sum must each remain within
`1e-6` of the dynamic frozen path.

## State-machine boundary

No Stage 3F-A output is referenced by `Path.pop`, `Path.push`, the search
queue, graph serialization, or the auxiliary branch decoder. This stage is
teacher-forced anchor validation, not closed-loop road-graph extraction.
