# Stage 2A independent trajectory encoder results

This directory contains the portable CPU and CUDA smoke reports for the
independent `TrajectoryFragmentEncoder`.

## Encoder

- Hidden dimension: 128
- Attention heads: 4
- Transformer layers: 2
- Dropout: 0.1
- Parameter count: 398,208
- Fragment pooling: CLS token
- Positional encoding: dynamic sinusoidal encoding
- Fragment interaction: none

Each point is represented by normalized XY, adjacent displacement, signed
log-scaled adjacent time delta, the inside-window flag, and the segment-only
flag. The encoder does not consume a learned `track_index` embedding.

## Real Xi'an smoke inputs

The smoke uses one real GT node from each category:

| Node type | GT vertex | Available fragments | Tracks | Points |
| --- | ---: | ---: | ---: | ---: |
| Ordinary | 332 | 131 | 117 | 682 |
| T junction | 414 | 47 | 41 | 260 |
| Multi-branch | 371 | 1,118 | 958 | 7,432 |

The fragment query uses a 256-pixel window, two context points, a 60-second
maximum time gap, and a 256-pixel maximum spatial gap.

## Smoke results

All point and fragment tokens are finite for every budget. Padded point and
fragment tokens are exactly zero.

| Budget | Point-token shape | Fragment-token shape | CPU median forward | CUDA median forward | CUDA peak allocated |
| ---: | --- | --- | ---: | ---: | ---: |
| 32 | `[3,32,9,128]` | `[3,32,128]` | 10.59 ms | 4.29 ms | 16.25 MiB |
| 64 | `[3,64,9,128]` | `[3,64,128]` | 13.57 ms | 7.32 ms | 21.11 MiB |
| 128 | `[3,128,9,128]` | `[3,128,128]` | 23.17 ms | 6.39 ms | 29.51 MiB |

These timings are smoke-only measurements on the local machine. They are not
formal performance benchmarks; the small CUDA workloads show normal scheduling
variation.

See `smoke_cpu.json` and `smoke_cuda.json` for complete counts, timing samples,
process RSS measurements, and tensor memory sizes.

## Reproduction

CPU:

```bash
python scripts/smoke_trajectory_encoder.py \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --graph data_self/input/graphs/xian.graph \
  --device cpu \
  --budgets 32 64 128 \
  --max-time-gap-seconds 60 \
  --max-spatial-gap-pixels 256 \
  --output data_self/output/stage2a_trajectory_encoder/smoke_cpu.json
```

CUDA:

```bash
python scripts/smoke_trajectory_encoder.py \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --graph data_self/input/graphs/xian.graph \
  --device cuda \
  --budgets 32 64 128 \
  --max-time-gap-seconds 60 \
  --max-spatial-gap-pixels 256 \
  --warmup-iterations 2 \
  --repeat-iterations 5 \
  --output data_self/output/stage2a_trajectory_encoder/smoke_cuda.json
```
