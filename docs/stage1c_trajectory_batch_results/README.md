# Stage 1C structured trajectory batch results

This directory contains the portable Stage 1C validation artifacts generated
from the real Xi'an structured trajectory cache and GT road graph.

## Inputs and settings

- Cache: `data_self/input/traj_structured/xian/v1`
- GT graph: `data_self/input/graphs/xian.graph`
- Cache basis:
  `trajectory_sample_points_and_segments_supercover_cells`
- Window size: 256 pixels
- Context points: 2
- Maximum time gap: 60 seconds
- Maximum spatial gap: 256 pixels
- Fragment budgets: 32, 64, 128, and 256
- Visualization budget: the 64 geometrically nearest fragments

The 8192x8192 background is top-left aligned zero padding of the original
4300x5000 image. No coordinate scaling or offset is applied.

## Real GT node coverage

The analysis covers all 639 selected GT road nodes:

| Node type | Nodes | Median fragments | Median tracks | Median points |
| --- | ---: | ---: | ---: | ---: |
| Ordinary (degree 2) | 369 | 876 | 774 | 4,658 |
| T junction (degree 3) | 214 | 689.5 | 628 | 4,194.5 |
| Multi-branch (degree >= 4) | 56 | 1,219 | 1,107.5 | 7,421.5 |
| Overall | 639 | 831 | 740 | 4,793 |

Candidate grid lookup has a median runtime of 0.123 ms. Exact fragment query
has a median runtime of 374.42 ms. The query uses the segment-aware grid and
does not scan all 14,267 tracks for every node.

Candidate-level truncation rates are 96.80%, 93.75%, 88.00%, and 77.79% for
budgets 32, 64, 128, and 256 respectively. These rates describe the size of
the high-recall candidate set, not trajectory quality.

See `gt_node_trajectory_batch_stats.json` for the complete distributions and
timing report.

## Visual validation

The cyan dashed lines are incident GT graph edges, the red star is the current
road node, and the colored polylines are the 64 retained fragments. Selection
uses only minimum polyline distance with deterministic track/index tie-breaks.
It does not use direction, frequency, density, or a fixed circle.

- `ordinary_vertex_0659_fragments_1009.png`
- `ordinary_vertex_0068_fragments_2257.png`
- `t_junction_vertex_0319_fragments_0691.png`
- `t_junction_vertex_0098_fragments_1963.png`
- `multi_branch_vertex_0300_fragments_1220.png`
- `multi_branch_vertex_0578_fragments_2223.png`

## Reproduction

```bash
python scripts/analyze_trajectory_batch_at_gt_nodes.py \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --graph data_self/input/graphs/xian.graph \
  --output-dir data_self/output/stage1c_gt_nodes \
  --background-image data_self/input/imagery_8192/xian.png \
  --max-time-gap-seconds 60 \
  --max-spatial-gap-pixels 256 \
  --budgets 32 64 128 256 \
  --visualization-cases-per-type 2 \
  --visualization-max-fragments 64
```
