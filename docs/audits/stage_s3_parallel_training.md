# Stage S3R remote parallel training

## Outcome

Formal execution completed on REMOTE_TRAINING_SERVER at run-code SHA 2e68f4e5a1c7cfad041182c2edce3194b8175b8c. All six controlled runs completed the full 102,400-step budget and passed result validation. Peak experiment-level parallelism was four, using NVIDIA GeForce RTX 4090 GPUs.

The local Windows attempt is retained as INVALID_FOR_FORMAL_TRAINING (LOCAL_EXECUTION_ENVIRONMENT_MISMATCH). The first remote attempt at 2e6cadab33dfe2de3a2a34f93339971f186d59f3 is INVALIDATED_BY_CODE_CHANGE; the formal refrozen run is 2e68f4e5a1c7cfad041182c2edce3194b8175b8c.

## Gates and fairness

- Production CUDA preflight: PASS for C0/C1/C2/C3/J0/J1.
- Road/junction shape: [B,1,64,64]; anchor shapes: [B,4,256,256].
- Padding/valid-mask crop and 8192x8192 tiled segmentation stitching: PASS.
- First-20 remote parity and first-100 formal-run identity: PASS.
- One code SHA, seed, split, sample plan, initialization, budget, precision, and checkpoint rule.
- No OOM, NaN, nonzero exit status, or stderr output.

## GPU schedule and budget

Initial eligible GPUs were physical 2, 5, 6, and 7, all RTX 4090. C0/C1/C2/C3 started concurrently; J0 followed C0 and J1 followed C1. External workloads later appeared on GPUs 6 and 7 and slowed C2/C3; they were not interrupted.

The 100-step probe measured 0.1576076702 seconds/step. FULL_BASELINE used 102,400 steps, 5,120-step evaluation intervals, and 20 checkpoints. Checkpoints remain remote; no checkpoint, raster, dataset, cache, TensorBoard event, or large log is intended for Git.
