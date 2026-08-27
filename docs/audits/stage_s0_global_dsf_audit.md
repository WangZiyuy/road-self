# VecRoad_self Seg-Raster Stage S0 global legacy DSF audit

## Executive conclusion

The repository contains no model class, module, configuration mode, or import
named `DelvMap`. The only DelvMap token is a comment in `lib/graph.py:108`.
The historical raster/image fusion implementation is
`model.DSFNet.Unet_multistage`, registered as `RPNet.DSF` and selected by the
string `TRAIN.MODEL=DSFNet`.

The legacy DSF path can complete a synthetic GPU training forward/backward, but
it is not a usable end-to-end train/inference contract:

- its road/junction/traj heads return probabilities but training treats road
  and junction as logits;
- segmentation inference returns 64x64 maps to a 256x256 stitcher;
- anchor inference deliberately omits the raster and fails in the first DSF
  convolution;
- its full-resolution anchor decoder ignores the recursive `next_step`, making
  all four full-resolution anchor channels exactly equal;
- 36,756,638 parameters are registered but not reached in a DSF training
  forward, 23,460,363 reached parameters have `grad=None`, and another
  1,572,864 reached parameters receive exactly zero gradient;
- the located epoch-40 and epoch-50 checkpoints have tensor-identical DSF and
  DSF-anchor parameters across that interval;
- the real Xi'an aerial/raster pair is 4096x4096 versus 4300x5000, the raster is
  ternary 0/128/255 rather than binary, and the configured `traj_test` directory
  is absent.

Accordingly, repairing DSF as the next production path is **NO_GO**. A new,
minimal raster-to-segmentation path is a reasonable later direction, but
implementation is **BLOCKED** until a deterministic raster/aerial registration
and value contract is established. No such model was implemented in S0.

## Git isolation and scope

The audited worktree is `${BASELINE_WORKTREE}` on `feat/seg-raster-only`,
created from verified `origin/master` commit
`13488c768d147b37632ffeefe4b62cb3e94b36ec`. The original dirty worktree
`${DIRTY_WORKTREE}` remained on `master`; no checkout, switch, stash, reset,
rebase, merge, or production edit was performed there. No trajectory feature
branch existed in `git branch -a`, so no trajectory-branch merge base could be
computed; the only verifiable committed common baseline was `origin/master`.

The initial commands, return codes, stdout, default remote, graph, branch list,
and worktree list are preserved in `artifacts/stage_s0_git_start.json`.

The source-provenance gate passed for the 12 required critical files. After
normalizing line endings, all 12 files in the two worktrees have identical
content. The raw SHA-256 differences are attributable only to LF/CRLF byte
encoding. `CURRENT_DIRTY_SNAPSHOT` identity covers only those 12 critical
files; it does not assert that any other untracked file in the dirty worktree
equals the committed baseline. See `artifacts/stage_s0_source_provenance.json`.

## Reproducible method

All production paths were read-only. The only additions are audit scripts,
tests, Markdown, and JSON. Principal commands were:

| Command | Return code | Key output |
|---|---:|---|
| `git ls-files` plus Windows `os.walk(maxdepth=4)` equivalent | 0 | All 521 tracked paths are present in the union inventory; missing tracked count 0. |
| structured equivalent of required `rg -n` patterns | 0 | 3,030 classified symbol hits before generated-report paths were excluded. |
| `python -m py_compile tools/audit/*.py tests/test_stage_s0_audit.py` | 0 | Audit scripts and audit tests compile. |
| `python tools/audit/audit_legacy_dsf_runtime.py --device cuda:0` | 0 | Five production-model synthetic cases passed. |
| `python tools/audit/audit_infer_contract_runtime.py --device cuda:0` | 0 | Origin stitch passed; DSF stitch and DSF anchor calls reproduced their expected production failures. |
| `python tools/audit/audit_legacy_dsf_checkpoints.py ...` | 0 | 1,362 checkpoint-like files inventoried; epoch40/50 compared tensor by tensor. |
| `python tools/audit/audit_raster_alignment.py ...` | 0 | Seven aerial and four raster files inspected; Xi'an blocker reproduced. |
| `python -m pytest tests/test_trajectory_mode.py -q` | 0 | 10 passed, 4 subtests passed. |
| `python -m pytest tests -q` | 0 | 242 passed, 9 warnings, 4 subtests passed. |

Exact commands, cwd, return codes, stdout, and stderr live in the corresponding
JSON artifacts. Runtime values below are not inferred from static code.

## Repository and symbol inventory

`artifacts/stage_s0_repository_inventory.json` is the union of:

1. every `git ls-files` path, at any depth; and
2. every physical file at depth <=4, excluding `.git` and `__pycache__`.

Every record contains `path`, `tracked`, `size`, `category`,
`generated_by_stage_s0`, `repository_file_origin`, DSF/raster/sequence
relevance flags, and `review_status`. All 12 forced-review files exist and are
marked `REVIEWED`. Deep tracked files are not lost to the requested find-depth
limit. Stage S0 artifacts, reports, `tools/audit/*`, the added audit test, and
the test cache are explicitly marked as generated and excluded from the count
of repository files that existed at audit start; none is classified as
preexisting dead/legacy code. Generated Stage S0 files are also excluded from
symbol matching to prevent audit evidence recursively matching itself.

The final dirty-worktree targeted search yielded 537 candidate code/config
files, not 537 confirmed relevant files. Review flags classify 63 as relevant
to at least one of DSF/raster/sequence and 474 as irrelevant to all three. The
inventory records these as
`dirty_untracked_candidate_code_count`,
`dirty_untracked_relevant_after_review_count`, and
`dirty_untracked_irrelevant_after_review_count`; 63 + 474 = 537 is enforced by
the generator and audit tests.

The symbol artifact classifies every matching line as definition, config, or
call/reference and assigns its file category. Sequence-only research files
found by the global search were inventoried and interface-reviewed, but not
modified or continued.

## Forced-file review

| File | Verified role and result |
|---|---|
| `model/DSFNet.py` | Defines `Unet_multistage`; two actual encoders, trajectory/image decoders, first co-attention, three sigmoid segmentation heads. The “three encoders” docstring is not implemented. |
| `model/model.py` | Canonical RPNet. Registers DSF/Transformer/legacy modules only when trajectory modules are enabled; branches on `origin`/`DSFNet`; contains both anchor decoders. |
| `model/model2.py` | Compatibility import layer only, not a second RPNet implementation. |
| `train.py` | Same mode resolver as infer; trajectory mode controls module construction; BCEWithLogits contract; no `traj_road` loss. |
| `infer.py` | DSF segmentation loads whole-region raster; anchor iteration requests sequence only and passes `traj_image=None`. |
| `utils/trajectory_mode.py` | Only `none` and `legacy_current` exist. `none+DSFNet` is rejected. No `raster_seg_only` or `legacy_dsf` mode exists. |
| `utils/OSMDataset.py` | `legacy_current` requests raster and sequence together, even for origin. |
| `utils/model_utils.py` | Crops aerial/raster with the same indices and divides both by 255; sequence filtering is a separate field. |
| `utils/tileloader.py` | Aerial and raster both use `load_rect`/axis swap, but source-shape equality is not checked. |
| `configs/default_self.yml` | `TRAIN.MODEL=origin`, `TRAIN.USE_TRAJ=True`, no `TRAJ.MODE`; resolves to `legacy_current`. |
| `data_self/gen_dataset.py` | Standalone hard-coded `D:/DataSet/multi_data_down` legacy generator; not the active VecRoad dataset path and not geographically self-describing. |
| `tests/test_trajectory_mode.py` | Tests mode helpers and guards only; it had no production DSF forward, inference, shape, or gradient coverage. |

Other directly relevant files include `utils/utils.py` (stitch-time sigmoid),
`utils/additional_methods.py` (visualization-time sigmoid), image-only baseline
tests/validators, checkpoint helpers, `loss_and_found.py` (commented dead DSF
experiments), and `lib/graph.py` (the lone DelvMap comment). Their hits and
classifications are in the two inventory artifacts.

## Real execution-path matrix

Only model names `origin` and `DSFNet`, and only trajectory modes `none` and
`legacy_current`, are implemented. The 12-row matrix covers four train
combinations (including the rejected one), three segmentation-inference paths,
three anchor-inference paths, unit tests, and checkpoint loading.

Key results:

- There is no raster-only path that avoids sequence loading.
- `legacy_current+origin` loads the raster and sequence, registers DSF, ignores
  raster/DSF, and sends the Transformer sequence feature to anchor.
- `legacy_current+DSFNet` executes both raster DSF and sequence Transformer in
  training; raster effects and sequence effects are therefore causally
  confounded.
- Both train and infer use the same mode resolver and similarly construct
  RPNet, but segmentation and anchor inference provide different modalities.
- `test=True` returns before sequence/anchor. Origin upsamples road/junction by
  four; DSF does not.

See `artifacts/stage_s0_execution_path_matrix.json` and the causal diagrams in
`docs/audits/stage_s0_model_dataflow.md`.

## Activation, loss, and output semantics

| Output | Origin | DSF | Training consumer | Inference/visualization |
|---|---|---|---|---|
| road | logits | probability (`Sigmoid`) | `binary_cross_entropy_with_logits` | `MapContainer` applies sigmoid; visualization applies sigmoid |
| junction | logits | probability (`Sigmoid`) | `binary_cross_entropy_with_logits` | same |
| traj | absent | probability (`Sigmoid`) | no loss | visualization applies sigmoid |
| anchor | logits | logits | `binary_cross_entropy_with_logits` | anchor inference applies sigmoid |
| anchor_lowrs | logits | logits | `binary_cross_entropy_with_logits` | visualization applies sigmoid |

For DSF road/junction, the head sigmoid is followed by BCEWithLogits' internal
sigmoid during training; train post-processing, infer stitching, and legacy
visualization can apply another explicit sigmoid. This is a contract failure,
not merely a naming issue.

Mathematical analysis (not a runtime observation): if `p=sigmoid(x)` is passed
as a logit, `sigmoid(p)` lies in `[0.5, 0.7310585786]`. The second sigmoid cannot
express a probability below 0.5 and changes both threshold meaning and
gradients. Actual output ranges are separately recorded in
`artifacts/stage_s0_runtime_audit.json` and
`artifacts/stage_s0_infer_contract_runtime.json`.

## Runtime and shape audit

Environment: CUDA device `cuda:0`, NVIDIA GeForce RTX 4050 Laptop GPU. Inputs
were batch 1, aerial `[1,3,256,256]`, raster `[1,1,256,256]`, walked path
`[1,1,64,64]`, and `NUM_TARGETS=4`.

| Case | Forward | Backward | road/junction | anchor/anchor_lowrs | traj_road |
|---|---|---|---|---|---|
| origin none train | PASS | PASS | `[1,1,64,64]` | `[1,4,256,256]` | absent |
| origin none eval | PASS | not requested | `[1,1,64,64]` | `[1,4,256,256]` | absent |
| origin legacy train | PASS | PASS | `[1,1,64,64]` | `[1,4,256,256]` | absent |
| DSF legacy train | PASS | PASS | `[1,1,64,64]` | `[1,4,256,256]` | `[1,1,64,64]` |
| DSF legacy eval | PASS | not requested | `[1,1,64,64]` | `[1,4,256,256]` | `[1,1,64,64]` |

Every recorded tensor includes dtype, device, min/max/mean, and finite ratio.
No monkeypatch or audit copy was used because the production model completed on
the available GPU. Importing `model.model` changed
`CUDA_LAUNCH_BLOCKING` from absent to `"1"`; `cudnn.benchmark` remained false.
`Unet_multistage.__init__` also creates an unregistered CUDA tensor `temp`, but
the forward does not use it.

Production inference probes then showed:

- origin `test=True`: 256x256 road/junction and stitch PASS;
- DSF `test=True`: 64x64 road/junction, followed by
  `ValueError: operands could not be broadcast together with shapes
  (256,256) (64,64) (256,256)`;
- DSF anchor with the actual infer argument contract: `TypeError` because
  `conv2d` receives `NoneType` as the trajectory raster.

## Anchor time-step validity

Origin train/eval and origin legacy train had non-zero 0-vs-1, 0-vs-2, and
0-vs-3 differences for both outputs. DSF results were:

| Mode | Output | 0 vs 1 max/mean | 0 vs 2 max/mean | 0 vs 3 max/mean |
|---|---|---:|---:|---:|
| train | anchor | 0 / 0 | 0 / 0 | 0 / 0 |
| train | anchor_lowrs | 19.7758 / 3.0283 | 40.8898 / 15.8538 | 72.7342 / 33.4799 |
| eval | anchor | 0 / 0 | 0 / 0 | 0 / 0 |
| eval | anchor_lowrs | 0.4930 / 0.1565 | 0.9398 / 0.2730 | 3.1175 / 2.2707 |

The cause is source-level: `model/model.py:503-512` reuses `fi_ca1` and the
same DSF skip tensors every step. `next_step` is used only by
`next_step_final` and recursive-slot updates, never by the DSF full-resolution
decoder.

## Parameter registration and gradients

| Training mode | Total/trainable | Registered, not reached | Reached, grad=None | Reached, exactly-zero grad |
|---|---:|---:|---:|---:|
| origin none | 32,463,260 | 0 | 0 | 0 |
| origin legacy | 164,276,735 | 114,555,139 | 0 | 0 |
| DSF legacy | 164,276,735 | 36,756,638 | 23,460,363 | 1,572,864 |

Important DSF legacy details:

- trajectory encoder: 18,849,984 reached, finite non-zero gradients;
- image U-Net: 39,398,336 reached, finite non-zero gradients;
- trajectory decoder: 20,547,200 reached, all `grad=None`;
- traj head: 74,689 reached, all `grad=None`;
- co-attention: 4,411,848 reached; 2,838,470 `grad=None`, 1,572,864 zero;
- `W_b`, `W_s`, and `W_t` are reached but their gradients are exactly zero;
- `ca_soft_down1` has non-zero gradients; `ca_info_down1` is reached but its
  returned `info1` is discarded, so its gradient is null;
- `sfw1..4` are read by the hook surface but are not used to form returned
  outputs and have `grad=None`;
- DSF anchor decoder, Transformer, and `fuse_module_traj` receive gradients;
- `cross_attention`, `traj_to_img_fc`, `stage_1_traj`,
  `stage_1_traj_aerial`, `missing_traj_feature`, origin Res2Net/fuse/decoder in
  DSF mode, and unused co-attention stages are registered but not reached.

Optimizer membership and checkpoint-key membership are separate recorded
columns and are not used as evidence of execution or training.

## Checkpoint audit

The read-only search across both worktrees found 1,362 `*.pth`, `*.pth.tar`,
`*.pt`, or `*.ckpt` files. None were copied or added to Git. The named historical
pair exists:

- `.../40.2047.pth.tar`: 1,090,456,446 bytes, metadata outer/path `40/2047`;
- `.../50.2047.pth.tar`: 1,090,456,446 bytes, metadata outer/path `50/2047`.

Both contain 1,175 state keys with identical key sets. Tensor-level comparison:

| Group | Tensors | Elements | Changed tensors/elements | Max abs diff |
|---|---:|---:|---:|---:|
| DSF | 368 | 86,260,670 | 0 / 0 | 0 |
| DSF anchor decoder | 74 | 20,579,018 | 0 / 0 | 0 |
| origin fuse_module | 36 | 3,451,376 | 0 / 0 | 0 |
| Transformer | 16 | 139,712 | 16 / 139,710 | 0.0719641 |
| fuse_module_traj | 36 | 3,792,720 | 36 / 3,792,544 | 0.256909 |
| Res2Net | 517 | 23,728,802 | 517 / 23,322,293 | 20,480 |

“在所比较的 epoch40/50 checkpoint 对中，全部 368 个 DSF tensors
和全部 74 个 DSF anchor decoder tensors 均逐元素完全相同；
Transformer、fuse_module_traj 和 Res2Net 等模块存在变化。”

Checkpoint source commit provenance is **UNKNOWN**. Relative to the current
model state dict, each checkpoint is missing 16 expected keys and has 28 extra
keys, so the pair is not fully compatible with the current model. This evidence
is limited to the compared epoch40/50 pair; it cannot establish behavior before
epoch 40 or establish that the current source's DSF was never trained.

## Raster data and registration

The audit parsed every config, dataset/loader call site, infer path, and legacy
generation path, then inspected assets from the original worktree read-only.

| Region file | Aerial | Raster | Result |
|---|---:|---:|---|
| `20_0_0.png` | 5465x4367 | 5465x4367 | shape pass; geospatial metadata unavailable |
| `518_0_0.png` | 5444x5485 | 5444x5485 | shape pass; geospatial metadata unavailable |
| `653_0_0.png` | 3279x4375 | 3279x4375 | shape pass; geospatial metadata unavailable |
| `xian_0_0.png` | 4096x4096 RGB | 4300x5000 grayscale | `DATA_ALIGNMENT_BLOCKER` |
| `xian_0_1/1_0/1_1.png` | present | absent | unpaired |

The Xi'an raster SHA-256 and full value census prove values `{0,128,255}`;
division by 255 in `utils/model_utils.py:923-926` produces
`{0,0.5019607843,1}`. No resize or stretch was found. Train uses the same region
argument, axis swap, coordinates, and window indices for both sources, but it
silently crops without verifying equal extents. Infer uses separate whole-image
PIL loaders, not the train TileCache. `data_self/input/traj_test` is configured
but absent.

PNG files expose no CRS, geotransform, pixel resolution/origin, or nodata
metadata. Same filenames or shapes therefore do not prove geographic
registration. No automatic resize or crop correction was attempted.

## Train/infer contract status

Mode resolution and constructor intent match. The material mismatches are:

1. training requests raster and sequence together; segmentation inference
   requests raster only; anchor inference requests sequence only;
2. DSF `test=True` returns H/4 probabilities while origin returns H logits;
3. DSF segmentation cannot be stitched at the configured crop size;
4. DSF anchor inference omits the required raster;
5. strict checkpoint handling depends on entry: train and explicit
   `CKPT_FILE` are strict, historical `TEST.CKPT` uses permissive
   `load_pretrained`;
6. train wraps DataParallel before load; infer loads before optional wrapping.

Overall status: **FAIL for DSF end-to-end**, **PASS for origin synthetic
contracts**, and **PARTIAL for checkpoint-load consistency**.

## Tests and coverage gaps

The previously captured full suite passed with 242 tests and remains recorded
in `artifacts/stage_s0_test_results.json`. During final bookkeeping, only
`tests/test_stage_s0_audit.py` was rerun: 12 passed with one cache-write warning
because `.pytest_cache` is not writable. The finalization run did not repeat
runtime, gradient, or checkpoint audits. Production-model GPU harness artifacts
provide the existing shape, gradient, anchor-step, and infer-failure evidence.

The pre-existing trajectory-mode tests covered helpers/guards but not DSF
forward, raster I/O, segmentation inference, anchor inference, or gradients.
The full coverage matrix is explicitly labelled as static-token classification;
runtime pass/fail comes only from captured commands.

## Historical claims, independently re-verified

| ID | Historical claim | Status | Evidence types | Result boundary |
|---:|---|---|---|---|
| 1 | No model named DelvMap; implementation is DSFNet/Unet_multistage | PASS | STATIC_CODE | Only a DelvMap comment exists; DSF model class verified. |
| 2 | DSF has raster/image encoders, co-attention, multiple heads | PASS | STATIC_CODE, RUNTIME | Two actual encoders, one executed co-attention stage, three segmentation heads. |
| 3 | Default is `MODEL=origin`, `USE_TRAJ=True` | PASS | CONFIG, TEST | Resolves to `legacy_current`. |
| 4 | Origin uses sequence-to-anchor, not raster-to-segmentation | PASS | STATIC_CODE, GRADIENT | Raster is loaded and DSF registered, but both are not reached. |
| 5 | DSF road/junction/traj heads contain Sigmoid | PASS | STATIC_CODE, RUNTIME | Runtime values are probabilities. |
| 6 | Training uses BCEWithLogits | PASS | STATIC_CODE | Applied to all supervised outputs, including already-sigmoided DSF road/junction. |
| 7 | Train or infer applies another Sigmoid | PASS | STATIC_CODE, RUNTIME | Train postprocess, infer stitch, and visualization do; BCEWithLogits adds its internal sigmoid. |
| 8 | DSF segmentation is input/4 | PASS | STATIC_CODE, RUNTIME | 256 input -> 64 road/junction/traj in train and test. |
| 9 | Anchor inference passes `traj_image=None` to DSF | PASS | STATIC_CODE, RUNTIME | Reproduced TypeError at first trajectory convolution. |
| 10 | DSF full-resolution anchor steps are identical | PASS | RUNTIME, TEST | Exact in train/eval; does not apply to anchor_lowrs. |
| 11 | DSF mode also enables sequence Transformer | PASS | STATIC_CODE, GRADIENT | Transformer and trajectory fuse are reached with gradients. |
| 12 | `traj_road` is absent from total loss | PASS | STATIC_CODE, GRADIENT | Decoder/head reached, all gradients null. |
| 13 | Many registered parameters have no effective forward/gradient | PASS | RUNTIME, GRADIENT | Quantified per mode and per parameter. |
| 14 | Epoch40/50 DSF checkpoint tensors do not change | PASS | CHECKPOINT | Exact tensor equality for the located pair only. |
| 15 | Xi'an raster values are 0/128/255 | PASS | DATA_FILE | Full unique-value census and checksum. |
| 16 | Xi'an raster is 4300x5000; aerial tile is 4096x4096 | PASS | DATA_FILE | Real files inspected, blocker recorded. |
| 17 | Required `traj_test` directory is absent | PASS | CONFIG, DATA_FILE | Configured path checked and absent. |

Detailed per-claim evidence objects, commands, line snippets, impacts, and
recommended actions are in `artifacts/stage_s0_conclusion.json`.

## Disproved or narrowed assumptions

- `model/model2.py` is not a second full RPNet; it is a compatibility import.
- The DSF docstring's “three encoders” is not the executed structure; there are
  trajectory and image encoders.
- DSF is not universally unable to run: production synthetic GPU forward and
  backward complete. Its formal inference contracts still fail.
- Not every DSF anchor output is time-invariant: full-resolution `anchor` is
  identical, while `anchor_lowrs` differs.
- Not all co-attention parameters are dead: `ca_soft_down1` gets non-zero
  gradients. `W_b/W_s/W_t` get zero gradients and `ca_info_down1` gets none.

## GO / NO-GO and next-stage proposal

`repair_legacy_dsf = NO_GO` as the primary engineering path. Its activation
semantics, train/infer modality mismatch, dormant parameter surface, unsupervised
decoder, non-recursive full anchor, and unchanged checkpoint tensors make repair
poorly bounded and hard to attribute.

`implement_new_raster_seg_only = BLOCKED` for immediate implementation because
the Xi'an data contract is not aligned and test raster data is missing. Once
that blocker is resolved, the proposed next stage is:

1. establish a manifest-driven, checksumed aerial/raster pairing contract with
   identical pixel grids, explicit ternary/binary semantics, and train/test
   coverage;
2. freeze an image-only baseline and define segmentation-only evaluation before
   any anchor experiment;
3. design a minimal raster+RGB segmentation encoder with logits-only road and
   junction heads, no sequence loader or Transformer, explicit H/4-to-H output
   contract, and unit-tested train/infer parity;
4. only after segmentation gains are demonstrated, expose a documented feature
   tensor to the existing anchor path and run causal ablations: image only,
   raster segmentation feature, prediction-only, and oracle segmentation;
5. keep the trajectory-sequence branch isolated until a later explicitly
   approved integration stage.

These are proposals only. S0 did not implement, repair, or integrate any model.
