# Stage S3B Final Report

## Phase A

Six of six formal LR-screen runs completed `20,480` optimizer steps and passed
integrity checks under formal training SHA
`02068550f2acd97de6ae44dc97ffde0ae72344ff`. Each run saved the nine required
versioned checkpoints at steps `0, 2560, ..., 20480`; all first-100-batch sample
identity and common-tensor checks passed. Execution was remote-only and used a
single eligible RTX 4090 in FIFO order; no external process was terminated.

LR selection used only image-only controls, as preregistered:

- `A0`, LR×1.0: best repair composite `0.054127`, retention `0.192760`.
- `A2`, LR×0.3: best repair composite `0.072230`, retention `0.327985`.
- `A4`, LR×0.1: best repair composite `0.076302`, retention `0.567801`.

Aligned runs `A1/A3/A5` were excluded from LR selection. LR×0.1 was the best
diagnostic candidate, but it did not pass the required retention threshold.
The image-only early-stop simulation selected from A4 would stop at step
`12800`, after its best repair composite at step `5120`.

## Training protocol repair

`FAIL`: retention stayed below `0.70` for every image-only LR candidate.
Lowering LR reduced deterioration but did not repair it sufficiently.

## Downstream phases

`NOT_EXECUTED_BY_LR_STABILITY_GATE`. No Phase B loss screen, Phase C
aligned/zero/shifted control matrix, anchor evaluation, or graph evaluation was
started. Balanced junction losses were audited but never trained, so no loss or
causal raster claim is made.

## Decision

Segmentation multi-seed: `NO_GO`; end-to-end multi-seed: `NO_GO`. No additional training was started.

The two pre-formal attempts (`b494a0d`, `8c0f50b`) remain
`INVALIDATED_BY_CODE_CHANGE` before any optimizer step. The failure-only reducer
is separately versioned at
`5357f44baf94b07051d25d3f5d1ab0a2ce87d144`; it does not alter formal training
provenance or any completed run.
