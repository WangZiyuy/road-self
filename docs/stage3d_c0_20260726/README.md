# road_self Stage 3D-C0

Frozen offline validation of support-guided trajectory context aggregation. No parameter was trained or modified.

## Branch comparison

| variant | branch AP | slot AP | endpoint mean px | direction mean deg | exact count | oracle duplicate | distinct coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original_attention | 0.9005 | 0.9092 | 5.02 | 6.09 | 0.7051 | 0.0000 | 0.8522 |
| no_trajectory | 0.8955 | 0.9025 | 5.65 | 6.59 | 0.7188 | 0.0000 | 0.8462 |
| support_aggregation | 0.9005 | 0.9092 | 5.00 | 5.87 | 0.7051 | 0.0000 | 0.8522 |
| support_topk_4 | 0.9012 | 0.9075 | 4.94 | 5.58 | 0.7051 | 0.0069 | 0.8462 |
| support_topk_8 | 0.9019 | 0.9083 | 4.94 | 5.59 | 0.7051 | 0.0069 | 0.8482 |
| support_topk_16 | 0.8972 | 0.9084 | 4.96 | 5.64 | 0.7051 | 0.0069 | 0.8441 |
| support_topk_32 | 0.9002 | 0.9089 | 4.97 | 5.69 | 0.7031 | 0.0069 | 0.8502 |
| random_aggregation (mean) | 0.8998 | 0.9083 | 5.06 | 6.29 | 0.7070 | 0.0000 | 0.8534 |

`full` is an explicit alias of frozen E4 `original_attention`.

## Branch AP by road category

| category | no trajectory | original | full support | support_topk_4 | support_topk_8 | support_topk_16 | support_topk_32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ordinary | 0.9872 | 0.9875 | 0.9876 | 0.9872 | 0.9873 | 0.9833 | 0.9876 |
| t_junction | 0.6245 | 0.6707 | 0.6703 | 0.6566 | 0.6493 | 0.6472 | 0.6554 |
| multi_branch | 0.7795 | 0.8091 | 0.8124 | 0.8154 | 0.8136 | 0.8159 | 0.8126 |

## Trajectory selection

- support AP: **0.7819**
- Precision@8 / Recall@8 / nDCG@8: **0.6374 / 0.3057 / 0.5872**
- query top-8 Jaccard median: **0.06666666666666667**
- query overlap by top-k: k=4 median=0.0, k=8 median=0.06666666666666667, k=16 median=0.14285714285714285, k=32 median=0.4883720930232558

## Context similarity

- original inter-query cosine mean: **0.9998828310940023**
- support inter-query cosine mean: **0.9496523955172599**
- aligned original/support cosine mean: **0.9664573821549615**
- by category aligned cosine: ordinary=0.9720357187037463, t_junction=0.9432504243320889, multi_branch=0.9492669540146986

## Decision

- support - original branch AP: **+0.0014**
- best support aggregation: **support_topk_8**
- full-support - original branch AP: **+0.0000**
- support - random mean branch AP: **+0.0022** (2.53 random-baseline std)
- full - no-trajectory branch AP: **+0.0050**
- best-support slot AP change: **-0.0009**
- best-support oracle distinct-coverage change: **-0.0040**
- best-support oracle duplicate ratio: **0.0069**
- best-support category AP changes: ordinary=-0.0002, t_junction=-0.0214, multi_branch=+0.0044
- enter next-stage training: **no**
- diagnosis: offline aggregation evidence is inconclusive and does not meet the configured full-versus-no-trajectory gain

The Stage 3D-A head reads final branch tokens and is therefore circular for online use. Its result is an offline diagnostic upper bound, not an implementation to feed into Path.push.

RPNet, E4, the support head, anchor, branch GT, compression, and Path.push remained unchanged.
