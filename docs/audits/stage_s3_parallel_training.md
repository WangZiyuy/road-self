# Stage S3 controlled parallel-training protocol

## Frozen scope

Stage S3 changes no fusion architecture. The only production-model semantic
extension makes `ANCHOR_GRAD_TO_SEG=false` apply to both `none` and
`raster_seg_only`; its default remains `true`, so the historical image-only
path is unchanged. The image backbone remains RGB-only. Raster data can reach
anchor prediction only through segmentation features.

Trajectory sequence input, Transformer, `fuse_module_traj`, legacy DSF,
DataParallel, DDP, and a direct raster-to-anchor path are forbidden.

## Six-run matrix

| Key | Input | Raster control | Anchor gradient to segmentation |
|---|---|---|---|
| C0 | image only | none | detached |
| C1 | image + raster | aligned | detached |
| C2 | image + raster | zero | detached |
| C3 | image + raster | fixed shift (+128,+128) | detached |
| J0 | image only | none | joint |
| J1 | image + raster | aligned | joint |

Every run uses seed `20260827`, 256×256 crops, four anchor targets, fp32,
teacher-forced path evolution, the same optimizer/loss weights, and the same
frozen spatial split. The shift is applied on the canonical full canvas with
zero fill; it never wraps. Zero and shifted controls retain the real valid
mask.

## Initialization and sample parity

No trusted image-only checkpoint compatible with the current S2 model was
available. A deterministic common initialization snapshot was generated once
and is excluded from Git. It contains image-only and raster-capable states:
all 648 shared tensors have identical keys, shapes, and values; the only 18
additional keys belong to `segmentation_raster_fusion`; the residual
`delta_projection` weight and bias are exactly zero.

The Xi’an 4096 training tile is split into disjoint left and right extents with
a 256-pixel excluded buffer. A crop must be fully contained within its extent.
The first 100 teacher-forced batches were regenerated independently for all
six configurations. Sample identities and every common aerial/walked-path/
road/junction/anchor target checksum matched. Raster control content differed
as declared, while valid-mask checksums matched.

## CUDA preflight and scheduling gate

Long training is prohibited until the frozen code commit is pushed and the
checkout is clean. Production preflight then requires a real Xi’an batch on
CUDA for all six modes, forward/loss/backward/optimizer update, finite logits,
losses and gradients, exact output shapes, anchor-channel diversity, detach/
joint gradient behavior, a padding-crossing crop, and a complete tiled
8192×8192 segmentation sweep. The same strict initialization loader and fp32
precision are used by preflight and training.

GPU discovery samples `nvidia-smi` three times at ten-second intervals. A GPU
is rejected for an external compute process, utilization over 15%, insufficient
free memory, an exclusion-list match, or a query error. The launcher never
kills or preempts a process. It uses the largest homogeneous eligible GPU pool,
one physical GPU per run, `CUDA_VISIBLE_DEVICES=<physical index>`, and FIFO
queueing when GPUs are fewer than jobs.

All live run products are under the Git-ignored
`${RUN_ROOT}`. Generated checkpoints, raster controls, memory maps, logs,
TensorBoard data, and datasets are never commit candidates.

## Code-freeze behavior

Each worker verifies branch, HEAD, and an empty Git status at startup and at
every metrics interval. A SHA mismatch or any code/config change marks the run
invalid. A code fix requires stopping and restarting the complete comparable
matrix from a new frozen commit.

## Executed gate outcome

The run code was frozen and pushed as
`2e6cadab33dfe2de3a2a34f93339971f186d59f3`. The checkout remained clean.
The production GPU gate then ran for the full declared 30-minute window and
completed 37 rounds, each containing three inventory samples. The host exposed
one NVIDIA GeForce RTX 4050 Laptop GPU. Every round found the same external
compute context (`BongoCat.exe`, PID 36908), so the GPU was ineligible even
when utilization and free memory otherwise appeared adequate.

Final preflight status is `BLOCKED_NO_ELIGIBLE_GPU`. No external process was
terminated or preempted. No Stage S3 training process was launched; peak
parallel jobs was zero. Consequently the production model step, memory sizing,
100-step budget probe, 8192 sweep, six-run training, and performance evaluation
were not executed. This is an infrastructure eligibility block, not a failed
model/data result.
