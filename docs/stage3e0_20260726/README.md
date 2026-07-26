# Stage 3E-0 trajectory evidence encoder

Stage 3E-0 inserts a branch-independent trajectory evidence pathway between
the frozen trajectory fragment encoder and the frozen E4 branch decoder:

```text
fragment tokens
  -> 4 learned trajectory latent queries cross-attend the fragment set
  -> 4 trajectory evidence tokens
  -> frozen E4 branch-to-trajectory cross-attention
  -> frozen E4 context fusion and branch heads
```

The latent queries do not receive branch queries, imagery, walked path, or
graph state. They represent slots for local trajectory structure rather than
road-branch slots.

## Experiment

- Teacher-forced split: 2048 train / 512 validation samples.
- Seed: 20260724.
- Fragment budget: bounded-near-diverse 64.
- E4 checkpoint:
  `data_self/stage3c_r2/e4/checkpoints/stage3c_aux.best.pth.tar`.
- Trainable module: `TrajectoryEvidenceEncoder` only, 67,072 parameters.
- Frozen: RPNet, graph-state encoder, trajectory fragment encoder, E4 branch
  decoder and output heads.
- Best epoch: 29/30.
- Training and final evaluation time after cache construction: 124.69 s.
- Peak allocated CUDA memory during the cached training loop: 0.177 GiB.

## Branch results

| trajectory input | Branch AP | endpoint px | direction deg | exact count | oracle coverage | oracle duplicate |
|---|---:|---:|---:|---:|---:|---:|
| image + graph, no trajectory | 0.895513 | 5.652 | 6.593 | 0.7188 | 0.8462 | 0.0000 |
| original fragment tokens | 0.900514 | 5.019 | 6.088 | 0.7051 | 0.8522 | 0.0000 |
| trajectory evidence tokens | **0.913859** | **4.894** | **5.782** | 0.7031 | **0.8563** | 0.0000 |

The evidence pathway improves Branch AP by:

- `+0.018346` over image + graph;
- `+0.013345` over the original fragment-token path.

## Results by node type

| trajectory input | ordinary AP | T-junction AP | multi-branch AP |
|---|---:|---:|---:|
| image + graph | 0.987188 | 0.624539 | 0.779534 |
| original fragments | 0.987483 | 0.670718 | 0.809277 |
| trajectory evidence | **0.988200** | **0.722478** | **0.849710** |

The largest gains occur on T-junction and multi-branch states.

## Evidence-token diagnosis

All tensors are finite, but the four latent tokens have almost completely
collapsed to the same representation:

| diagnostic | mean | median | P90 |
|---|---:|---:|---:|
| evidence-token pairwise cosine | 0.999341 | 0.999432 | 0.999707 |
| fragment-attention pairwise cosine | 0.999856 | 0.999878 | 0.999990 |
| latent-query top-8 fragment Jaccard | 0.993056 | 1.000000 | 1.000000 |
| normalized attention entropy | 0.931187 | 0.936375 | 0.966326 |

Therefore, the positive branch result validates a useful independent
trajectory evidence pathway, but it does **not** validate four distinct
trajectory-structure slots. In this experiment the module behaves much more
like one useful global trajectory-set summary repeated four times.

No diversity loss, branch conditioning, joint fine-tuning, support
replacement, anchor fusion, or `Path.push` change was introduced to repair
this collapse in Stage 3E-0.

## Artifacts

- `evaluation.json`: final best-checkpoint metrics.
- `evaluation_recheck.json`: independent strict checkpoint reload plus the
  extended attention-collapse diagnostics.
- `training_curve.jsonl`: all 30 epochs.
- `training_summary.json`: complete run metadata and validation records.
- `fragment_attention.npz`: per-sample latent-to-fragment attention arrays.
- `visualizations/`: ordinary, T-junction, and multi-branch examples.

Best checkpoint on the 237 server:

```text
/home/wangziyu/VecRoad_self/data_self/stage3e0/trajectory_evidence/checkpoints/stage3e0.best.pth.tar
SHA256: 82aec2080d812129f167f1798899c31017772d6cee0f9b68d1f9255ef8109f63
```

## Commands

```bash
python train_trajectory_evidence.py \
  --config configs/stage3e0_trajectory_evidence.yml \
  --mode train

python train_trajectory_evidence.py \
  --config configs/stage3e0_trajectory_evidence.yml \
  --mode evaluate \
  --checkpoint data_self/stage3e0/trajectory_evidence/checkpoints/stage3e0.best.pth.tar \
  --output-dir data_self/stage3e0/trajectory_evidence_recheck
```

Validation:

```text
11 Stage 3E-0 tests passed.
199 repository tests passed.
```
