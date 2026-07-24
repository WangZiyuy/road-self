# Stage 3C-R2 formal E0-E4 comparison

All five experiments use the same 2048/512 spatial split, seed, frozen strict RPNet checkpoint, 30 epochs, optimizer settings, trajectory dropout 0.25, and bounded trajectory budget 64. Every auxiliary run starts fresh.

Summed training time: 916.2 seconds; peak CUDA memory was approximately 2.82 GiB.

| Experiment | Best epoch | Full AP | No-traj AP | Trajectory+graph AP | Slot AP | Exact count | Distinct coverage | Oracle duplicate | Matched duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 | 30 | 0.1111 | 0.1047 | 0.0646 | 0.2345 | 0.2422 | 0.5324 | 0.4690 | 0.3655 |
| E1 | 29 | 0.1140 | 0.1075 | 0.0532 | 0.2391 | 0.2422 | 0.7267 | 1.0000 | 1.0000 |
| E2 | 15 | 0.0879 | 0.0841 | 0.0324 | 0.2560 | 0.1816 | 0.6559 | 1.0000 | 1.0000 |
| E3 | 25 | 0.1003 | 0.1207 | 0.0315 | 0.2538 | 0.1816 | 0.6802 | 1.0000 | 0.9724 |
| E4 | 30 | 0.9005 | 0.8955 | 0.8493 | 0.9092 | 0.7051 | 0.8522 | 0.0000 | 0.0069 |

## Full-modality diagnostics

| Experiment | Matched probability | Unmatched probability | Missed rate | Extra rate | Graph-query cosine | Final-query cosine |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 0.2259 | 0.1583 | 1.0000 | 0.0000 | 0.9996 | 0.9999 |
| E1 | 0.1920 | 0.1283 | 1.0000 | 0.0000 | 0.9996 | 0.9999 |
| E2 | 0.6015 | 0.4735 | 0.7267 | 0.8907 | 0.9996 | 0.9999 |
| E3 | 0.4725 | 0.3342 | 0.8806 | 0.9152 | 0.9996 | 0.9999 |
| E4 | 0.9182 | 0.1039 | 0.0304 | 0.3859 | 0.4168 | 0.5031 |

## Best experiment grouped by GT branch count

Best experiment: **E4**.

| GT group | Samples | Branch AP | Exact count | Distinct coverage | Oracle duplicate |
|---|---:|---:|---:|---:|---:|
| count_0 | 124 | 0.0000 | 0.5323 | 0.0000 | 0.0000 |
| count_1 | 317 | 0.9875 | 0.9180 | 0.9464 | 0.0000 |
| count_2 | 39 | 0.6707 | 0.0513 | 0.6026 | 0.0000 |
| count_ge_3 | 32 | 0.8093 | 0.0625 | 0.7475 | 0.0000 |

## Decisions

1. Existence matching alone effective? **No** (AP delta +0.0029)
2. No-object weighting alone effective? **No** (AP delta -0.0232)
3. Matching + no-object most stable? **No**
4. Self-attention improves validation? **Yes** (E4-E3 AP +0.8002)
5. Full stably better than no-trajectory? **No**
6. Multi-branch query collapse remains? **No**
7. Ready for trajectory-support supervision? **Yes**
8. Ready for anchor fusion? **No**

Best validation experiment: **E4**.

Per-modality, GT-count-grouped, probability, duplicate, and query-representation statistics are preserved in `comparison.json` and each experiment directory.

No branch endpoint was passed to `Path.push`; RPNet, anchor, trajectory encoding, sampling, branch GT, and decoder structure were not changed.
