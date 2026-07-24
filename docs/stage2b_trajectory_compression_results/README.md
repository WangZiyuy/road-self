# Stage 2B trajectory candidate compression results

This directory contains portable validation artifacts for the deterministic,
non-learned compression step inserted between Stage 1B fragment retrieval and
the Stage 1C batch builder.

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
- `near_diverse` nearest fraction: 0.25
- GT evaluation probe distance: 40 pixels
- GT evidence threshold: 20 pixels

GT information is used only for evaluation after selection. It is not an
input to either compression strategy.

The 8192x8192 background is top-left aligned zero padding of the original
4300x5000 image. No coordinate scaling or offset is applied.

## Real GT-node evaluation

The analysis covers all 639 selected GT road nodes and 1,607 incident GT
branches:

- 369 ordinary degree-2 nodes
- 214 T junctions
- 56 degree-4-or-higher multi-branch junctions

| Budget | Nearest coverage | Near-diverse coverage | Nearest mean time | Near-diverse mean time |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 89.05% | 92.72% | 1.99 ms | 3.52 ms |
| 64 | 90.85% | 93.03% | 3.15 ms | 6.04 ms |
| 128 | 92.16% | 93.03% | 5.20 ms | 10.73 ms |
| 256 | 92.78% | 93.03% | 8.58 ms | 18.89 ms |

At budget 64, the mean normalized nearest-position pairwise RMS increases
from 0.207 to 0.852, and the mean continuous road-axis dispersion increases
from 0.320 to 0.654. This confirms that `near_diverse` retains more spatial
and directional evidence than the nearest-distance baseline.

The complete run took 359.81 seconds, including 639 fragment queries,
descriptor construction, both strategies at four budgets, GT evaluation, and
six comparison figures. Exact fragment query time averaged 265.25 ms per
node. Descriptor construction averaged 176.93 ms per node.

See `compression_summary.json` for aggregate distributions and per-node-type
coverage. The full 36.5 MB per-node report remains under
`data_self/output/stage2b_trajectory_compression` in the local and 237
workspaces.

## Visual comparison

Each figure contains:

1. every Stage 1B high-recall candidate;
2. the 64 nearest candidates;
3. the 64 `near_diverse` representatives.

Cyan dashed lines are incident GT edges, yellow diamonds are evaluation probe
points, the green star is the current node, and the number printed on each
representative is its `support_count`.

- `multi_branch_vertex_0118_all_nearest64_diverse64.png`
- `multi_branch_vertex_0307_all_nearest64_diverse64.png`
- `multi_branch_vertex_0489_all_nearest64_diverse64.png`
- `multi_branch_vertex_0666_all_nearest64_diverse64.png`
- `t_junction_vertex_0092_all_nearest64_diverse64.png`
- `t_junction_vertex_0116_all_nearest64_diverse64.png`

These figures also show why this step remains a compute-budget compressor
rather than a final quality selector: continuous geometric diversity can
retain unusual or noisy geometry. Reliability must be learned later instead
of being replaced here by fixed direction bins or trajectory-frequency
thresholds.

## Reproduction

```bash
python scripts/analyze_trajectory_compression.py \
  --cache-dir data_self/input/traj_structured/xian/v1 \
  --graph data_self/input/graphs/xian.graph \
  --output-dir data_self/output/stage2b_trajectory_compression \
  --background-image data_self/input/imagery_8192/xian.png \
  --max-time-gap-seconds 60 \
  --max-spatial-gap-pixels 256 \
  --budgets 32 64 128 256 \
  --visualization-count 6
```

The Stage 0 through Stage 2B regression suite contains 72 tests and passes in
the local environment. The two Stage 2B-specific test modules contain 14
tests and also pass in the 237 `pytorch` environment.
