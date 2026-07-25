# road_self Stage 3D-A

Stage 3D-A learns a branch-conditioned fragment support score as an evaluation-only side head. It does not replace E4 trajectory attention and is not connected to anchor prediction or Path.push.

## Label diagnostics

- Overall support availability: **0.9446**
- Multi-branch support availability: **0.9499**
- Bounded-64 oracle support recall: **0.9446**
- Segment-only positive ratio: **0.0000**
- The configured 80% multi-branch availability gate: **passed**

## 32-sample overfit

- Support AP: 0.5603 -> **0.9347**
- Raw attention support AP: **0.6084**
- Reducible-loss reduction: **0.5662**
- Result: **passed**

## Validation support metrics

- Support-head AP / AUROC: **0.7819 / 0.8358**
- Raw attention AP / AUROC: **0.3910 / 0.4660**
- Recall@1/4/8/16: **0.0535 / 0.1734 / 0.3057 / 0.5108**
- Top-8 branch-pair Jaccard mean / median: **0.2778 / 0.0667**
- Best epoch: **35**
- Recall@K is the fraction of all positive fragments recovered within the top K, not an at-least-one-hit rate.
- The real validation cache contains no segment-only positive pair. Segment-only support is therefore code-path tested, but not empirically validated by this run.
- Visual inspection shows useful branch conditioning overall, but individual multi-branch queries can still rank low-target fragments in their top eight; the support head is not perfect.

## E4 modality ablation

| modality | branch AP |
| --- | ---: |
| graph_only | 0.8383 |
| trajectory_graph | 0.8493 |
| no_trajectory | 0.8955 |
| full | 0.9005 |

- trajectory_graph - graph_only: **+0.0109 AP**
- full - no_trajectory: **+0.0050 AP**

## Three-seed E4 stability

- Full AP mean/std: **0.9070 / 0.0065**
- No-trajectory AP mean/std: **0.9028 / 0.0074**
- Full-minus-no-trajectory mean/std: **0.0042 / 0.0010**
- Oracle duplicate mean/std: **0.0046 / 0.0065**
- Distinct coverage mean/std: **0.8610 / 0.0075**

## Decision

Stage 3D-A acceptance: **passed**.
This result validates an independent support head only. It does not authorize replacing trajectory cross-attention, adding a reliability loss, or connecting branch/support outputs to anchor or Path.push.

Best support-head checkpoint: `/home/wangziyu/VecRoad_self/data_self/stage3d_a/checkpoints/stage3d_support.best.pth.tar`.

Reproduction commands (from the road_self root on the 237 server):

```bash
python train_trajectory_support.py \
  --config configs/stage3d_a_support.yml --mode labels
python train_trajectory_support.py \
  --config configs/stage3d_a_support.yml --mode train \
  --device cuda
python scripts/run_stage3d_e4_stability.py \
  --config configs/stage3d_a_support.yml --device cuda
```

Tests: **158 passed**.
