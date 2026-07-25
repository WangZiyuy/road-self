# road_self Stage 3D-B0

Stage 3D-B0 tests a branch-conditioned selector before E4 trajectory cross-attention. The trainable branch input is exactly `concat(graph_conditioned_query, image/walked_path_context)`.

## Label diagnostics

- bounded-64 branch support hit rate: **0.9446**
- segment-only fragments inspected: **809**; positive in the unchanged label configuration: **0**

## Main comparison

| source | AP | AUROC | P@8 | Hit@8 | mass recall@8 | nDCG@8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| raw attention | 0.3910 | 0.4660 | 0.3895 | 0.8414 | 0.1590 | 0.2734 |
| post-fusion support | 0.7819 | 0.8358 | 0.6374 | 0.9789 | 0.3236 | 0.5872 |
| pre-trajectory support (3-seed mean) | 0.7819 | — | 0.6397 | 0.9774 | 0.3248 | 0.5880 |

Pre-trajectory AP mean/std: **0.7819 / 0.0020**.

## AP by GT branch count

| source | count=1 | count=2 | count>=3 | count>=2 |
| --- | ---: | ---: | ---: | ---: |
| raw attention | 0.4050 | 0.3708 | 0.3679 | 0.3682 |
| post-fusion support | 0.7934 | 0.7934 | 0.7331 | 0.7604 |
| pre-trajectory support | 0.7970 | 0.7783 | 0.7253 | 0.7486 |

## Separation and support-invalid branches

- predicted top-8 Jaccard median, 3-seed mean: **0.0667**
- GT-positive-set Jaccard median: **0.5263**
- support-invalid top-1 probability mean / p90: **0.5688 / 0.7700**

## Decision

- multibranch_raw_attention_ap_gain: **passed**
- pre_support_ap: **passed**
- raw_attention_ap_gain: **passed**
- support_invalid_not_generally_high: **failed**
- three_seed_stability: **passed**
- top_k_branch_separation: **passed**

Stage 3D-B0 acceptance: **failed**.
Support-guided aggregation: **not recommended; stop at Stage 3D-B0**.

No support score changed E4 trajectory attention, branch outputs, anchor prediction, or Path.push.

Tests: **166 passed**.
