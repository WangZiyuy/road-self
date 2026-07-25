# road_self Stage 3D-C0

Frozen offline validation of support-guided trajectory context aggregation. No parameter was trained or modified.

## Branch comparison

| variant | branch AP | endpoint mean px | direction mean deg | exact count | oracle duplicate | distinct coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| original_attention | 0.9005 | 5.02 | 6.09 | 0.7051 | 0.0000 | 0.8522 |
| no_trajectory | 0.8955 | 5.65 | 6.59 | 0.7188 | 0.0000 | 0.8462 |
| support_aggregation | 0.9005 | 5.00 | 5.87 | 0.7051 | 0.0000 | 0.8522 |
| random_aggregation (mean) | 0.8998 | 5.06 | 6.29 | 0.7070 | 0.0000 | 0.8534 |

`full` is an explicit alias of frozen E4 `original_attention`.

## Trajectory selection

- support AP: **0.7819**
- Precision@8 / Recall@8 / nDCG@8: **0.6374 / 0.3057 / 0.5872**
- query top-8 Jaccard median: **0.06666666666666667**

## Context similarity

- original inter-query cosine mean: **0.9998828310940023**
- support inter-query cosine mean: **0.9496523955172599**
- aligned original/support cosine mean: **0.9664573823974933**

## Decision

- support - original branch AP: **+0.0000**
- support - random mean branch AP: **+0.0008** (0.89 random-baseline std)
- full - no-trajectory branch AP: **+0.0050**
- enter next-stage training: **no**
- diagnosis: support ranking is strong but frozen aggregation does not materially improve branch AP; support labels are not changed, and the branch-token/decoder fusion plus post-fusion circular dependency must be analyzed before any integration

The Stage 3D-A head reads final branch tokens and is therefore circular for online use. Its result is an offline diagnostic upper bound, not an implementation to feed into Path.push.

RPNet, E4, the support head, anchor, branch GT, compression, and Path.push remained unchanged.
